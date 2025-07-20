#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <stdint.h>

/* Magic marker constants (must match encoder) */
#define FLAC__ENTROPY_CODING_START_MAGIC 0xDEADBEEF
#define FLAC__ENTROPY_CODING_END_MAGIC   0xCAFEBABE

/* FLAC constants */
#define FLAC__ENTROPY_CODING_METHOD_TYPE_LEN 2
#define FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_ORDER_LEN 4
#define FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_PARAMETER_LEN 4
#define FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE2_PARAMETER_LEN 5
#define FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_RAW_LEN 5
#define FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_ESCAPE_PARAMETER 15
#define FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE2_ESCAPE_PARAMETER 31

/* Simple bit reader structure */
typedef struct {
    const uint8_t *data;
    size_t size;
    size_t byte_pos;
    int bit_pos;
} BitReader;

/* Initialize bit reader */
void bitreader_init(BitReader *br, const uint8_t *data, size_t size) {
    br->data = data;
    br->size = size;
    br->byte_pos = 0;
    br->bit_pos = 0;
}

/* Read n bits from bit reader */
uint32_t bitreader_read_bits(BitReader *br, int n) {
    uint32_t result = 0;
    
    for (int i = 0; i < n; i++) {
        if (br->byte_pos >= br->size) {
            fprintf(stderr, "Error: Unexpected end of data\n");
            return 0;
        }
        
        uint8_t bit = (br->data[br->byte_pos] >> (7 - br->bit_pos)) & 1;
        result = (result << 1) | bit;
        
        br->bit_pos++;
        if (br->bit_pos == 8) {
            br->bit_pos = 0;
            br->byte_pos++;
        }
    }
    
    return result;
}

/* Read Rice-coded signed integer */
int32_t bitreader_read_rice_signed(BitReader *br, int parameter) {
    /* Read unary part (quotient) with bounds checking */
    uint32_t quotient = 0;
    const uint32_t MAX_QUOTIENT = 65535; // Reasonable upper bound to prevent infinite loops
    
    while (quotient < MAX_QUOTIENT && bitreader_read_bits(br, 1) == 0) {
        quotient++;
    }
    
    /* Check if we hit the maximum quotient (likely corrupted data) */
    if (quotient >= MAX_QUOTIENT) {
        fprintf(stderr, "Warning: Rice quotient exceeded maximum (%u), data may be corrupted\n", MAX_QUOTIENT);
        return 0; // Return safe default value
    }
    
    /* Read binary part (remainder) */
    uint32_t remainder = 0;
    if (parameter > 0) {
        remainder = bitreader_read_bits(br, parameter);
    }
    
    /* Combine quotient and remainder */
    uint32_t value = (quotient << parameter) | remainder;
    
    /* Convert to signed (Rice coding uses sign-magnitude representation) */
    if (value & 1) {
        return -((int32_t)(value >> 1)) - 1;
    } else {
        return (int32_t)(value >> 1);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <entropy_file> <output_file> <num_samples>\n", argv[0]);
        return 1;
    }
    
    const char *entropy_filename = argv[1];
    const char *output_filename = argv[2];
    uint32_t num_samples = atoi(argv[3]);
    
    if (num_samples == 0) {
        fprintf(stderr, "Error: Invalid number of samples: %s\n", argv[3]);
        return 1;
    }
    
    /* Read entropy-coded file */
    FILE *entropy_file = fopen(entropy_filename, "rb");
    if (!entropy_file) {
        fprintf(stderr, "Error: Cannot open entropy file %s\n", entropy_filename);
        return 1;
    }
    
    /* Get file size */
    struct stat st;
    if (stat(entropy_filename, &st) != 0) {
        fprintf(stderr, "Error: Cannot stat entropy file\n");
        fclose(entropy_file);
        return 1;
    }
    
    /* Read entire entropy file into memory */
    uint8_t *entropy_data = (uint8_t *)malloc(st.st_size);
    if (!entropy_data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(entropy_file);
        return 1;
    }
    
    if (fread(entropy_data, 1, st.st_size, entropy_file) != st.st_size) {
        fprintf(stderr, "Error: Failed to read entropy file\n");
        free(entropy_data);
        fclose(entropy_file);
        return 1;
    }
    fclose(entropy_file);
    
    /* Validate minimum file size for metadata */
    if (st.st_size < 1) {
        fprintf(stderr, "Error: File too small to contain entropy data\n");
        free(entropy_data);
        return 1;
    }
    
    fprintf(stderr, "Reading pure entropy-coded data (%zu bytes)\n", st.st_size);
    
    /* Initialize bit reader for metadata and Rice data (entire file) */
    BitReader br;
    bitreader_init(&br, entropy_data, st.st_size);
    
    /* Parse metadata */
    uint32_t method_type = bitreader_read_bits(&br, FLAC__ENTROPY_CODING_METHOD_TYPE_LEN);
    uint32_t partition_order = bitreader_read_bits(&br, FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_ORDER_LEN);
    
    fprintf(stderr, "Parsed metadata: method_type=%u, partition_order=%u\n", method_type, partition_order);
    
    /* Determine parameter length based on method type */
    int plen = (method_type == 1) ? FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE2_PARAMETER_LEN : 
                                   FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_PARAMETER_LEN;
    uint32_t pesc = (method_type == 1) ? FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE2_ESCAPE_PARAMETER : 
                                        FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_ESCAPE_PARAMETER;
    
    /* Allocate output buffer */
    int32_t *decoded_residuals = (int32_t *)malloc(num_samples * sizeof(int32_t));
    if (!decoded_residuals) {
        fprintf(stderr, "Error: Failed to allocate output buffer\n");
        free(entropy_data);
        return 1;
    }
    
    /* Decode residuals */
    uint32_t total_decoded = 0;
    uint32_t num_partitions = 1u << partition_order;
    uint32_t predictor_order = 0;  /* LPC order 0 */
    
    fprintf(stderr, "Decoding %u samples in %u partition(s)\n", num_samples, num_partitions);
    
    if (partition_order == 0) {
        /* Single partition */
        uint32_t rice_parameter = bitreader_read_bits(&br, plen);
        fprintf(stderr, "Rice parameter: %u\n", rice_parameter);
        
        if (rice_parameter < pesc) {
            /* Normal Rice coding */
            for (uint32_t i = 0; i < num_samples; i++) {
                decoded_residuals[i] = bitreader_read_rice_signed(&br, rice_parameter);
            }
            total_decoded = num_samples;
        } else {
            /* Raw coding */
            uint32_t raw_bits = bitreader_read_bits(&br, FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_RAW_LEN);
            fprintf(stderr, "Using raw coding with %u bits per sample\n", raw_bits);
            
            for (uint32_t i = 0; i < num_samples; i++) {
                uint32_t unsigned_val = bitreader_read_bits(&br, raw_bits);
                /* Convert to signed */
                if (raw_bits < 32 && unsigned_val >= (1u << (raw_bits - 1))) {
                    decoded_residuals[i] = (int32_t)(unsigned_val - (1u << raw_bits));
                } else {
                    decoded_residuals[i] = (int32_t)unsigned_val;
                }
            }
            total_decoded = num_samples;
        }
    } else {
        /* Multiple partitions */
        uint32_t default_partition_samples = (num_samples + predictor_order) >> partition_order;
        uint32_t sample_idx = 0;
        
        for (uint32_t p = 0; p < num_partitions; p++) {
            uint32_t partition_samples = default_partition_samples;
            if (p == 0) {
                partition_samples -= predictor_order;  /* First partition excludes warmup samples */
            }
            
            uint32_t rice_parameter = bitreader_read_bits(&br, plen);
            fprintf(stderr, "Partition %u: %u samples, rice_parameter=%u\n", p, partition_samples, rice_parameter);
            
            if (rice_parameter < pesc) {
                /* Normal Rice coding */
                for (uint32_t i = 0; i < partition_samples; i++) {
                    decoded_residuals[sample_idx + i] = bitreader_read_rice_signed(&br, rice_parameter);
                }
            } else {
                /* Raw coding */
                uint32_t raw_bits = bitreader_read_bits(&br, FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_RAW_LEN);
                
                for (uint32_t i = 0; i < partition_samples; i++) {
                    uint32_t unsigned_val = bitreader_read_bits(&br, raw_bits);
                    /* Convert to signed */
                    if (raw_bits < 32 && unsigned_val >= (1u << (raw_bits - 1))) {
                        decoded_residuals[sample_idx + i] = (int32_t)(unsigned_val - (1u << raw_bits));
                    } else {
                        decoded_residuals[sample_idx + i] = (int32_t)unsigned_val;
                    }
                }
            }
            
            sample_idx += partition_samples;
        }
        total_decoded = sample_idx;
    }
    
    fprintf(stderr, "Successfully decoded %u samples\n", total_decoded);
    
    /* Write decoded residuals to output file */
    FILE *output_file = fopen(output_filename, "wb");
    if (!output_file) {
        fprintf(stderr, "Error: Cannot open output file %s\n", output_filename);
        free(decoded_residuals);
        free(entropy_data);
        return 1;
    }
    
    if (fwrite(decoded_residuals, sizeof(int32_t), total_decoded, output_file) != total_decoded) {
        fprintf(stderr, "Error: Failed to write decoded samples\n");
        fclose(output_file);
        free(decoded_residuals);
        free(entropy_data);
        return 1;
    }
    
    fclose(output_file);
    
    /* Clean up */
    free(decoded_residuals);
    free(entropy_data);
    
    fprintf(stderr, "Decoding completed successfully\n");
    return 0;
} 