#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// FLAC type definitions
typedef int32_t FLAC__int32;
typedef uint32_t FLAC__uint32;
typedef uint64_t FLAC__uint64;
typedef int FLAC__bool;
#define true 1
#define false 0

// FLAC decoder configuration constants
#define FLAC_DEFAULT_PREDICTOR_ORDER 0              // No prediction for simplicity

// FLAC constants from format.h
#define FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_ESCAPE_PARAMETER 15
#define FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE2_ESCAPE_PARAMETER 31
#define FLAC__ENTROPY_CODING_METHOD_TYPE_PARTITIONED_RICE 0
#define FLAC__ENTROPY_CODING_METHOD_TYPE_PARTITIONED_RICE2 1
#define FLAC__ENTROPY_CODING_METHOD_TYPE_LEN 2
#define FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_ORDER_LEN 4
#define FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_PARAMETER_LEN 4
#define FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE2_PARAMETER_LEN 5
#define FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_RAW_LEN 5

// FLAC utility macros
#define flac_max(a,b) ((a) > (b) ? (a) : (b))
#define FLAC__ASSERT(x) 

// FLAC structures
typedef struct {
    FLAC__uint32 *parameters;
    FLAC__uint32 *raw_bits;
} FLAC__EntropyCodingMethod_PartitionedRiceContents;

typedef struct {
    uint32_t order;
    const FLAC__EntropyCodingMethod_PartitionedRiceContents *contents;
} FLAC__EntropyCodingMethod_PartitionedRice;

typedef struct {
    int type;
    FLAC__EntropyCodingMethod_PartitionedRice partitioned_rice;
} FLAC__EntropyCodingMethod;

// Simple BitReader implementation
typedef struct {
    const uint8_t *buffer;
    size_t buffer_size;
    size_t byte_idx;
    uint32_t bit_idx;
    uint32_t accumulator;
    uint32_t bits_in_accumulator;
} BitReader;

static BitReader* bitreader_new(const uint8_t *buffer, size_t size) {
    BitReader *br = malloc(sizeof(BitReader));
    if (!br) return NULL;
    
    br->buffer = buffer;
    br->buffer_size = size;
    br->byte_idx = 0;
    br->bit_idx = 0;
    br->accumulator = 0;
    br->bits_in_accumulator = 0;
    return br;
}

static void bitreader_delete(BitReader *br) {
    if (br) {
        free(br);
    }
}

static FLAC__bool bitreader_read_raw_uint32(BitReader *br, uint32_t *val, uint32_t bits) {
    if (bits == 0) {
        *val = 0;
        return true;
    }
    
    *val = 0;
    
    while (bits > 0) {
        // Fill accumulator if needed
        while (br->bits_in_accumulator < 8 && br->byte_idx < br->buffer_size) {
            br->accumulator |= ((uint32_t)br->buffer[br->byte_idx]) << (24 - br->bits_in_accumulator);
            br->bits_in_accumulator += 8;
            br->byte_idx++;
        }
        
        if (br->bits_in_accumulator == 0) {
            return false; // End of stream
        }
        
        uint32_t bits_to_read = (bits < br->bits_in_accumulator) ? bits : br->bits_in_accumulator;
        uint32_t mask = (1u << bits_to_read) - 1;
        
        *val <<= bits_to_read;
        *val |= (br->accumulator >> (32 - bits_to_read)) & mask;
        
        br->accumulator <<= bits_to_read;
        br->bits_in_accumulator -= bits_to_read;
        bits -= bits_to_read;
    }
    
    return true;
}

static FLAC__bool bitreader_read_raw_int32(BitReader *br, int32_t *val, uint32_t bits) {
    uint32_t uval;
    if (!bitreader_read_raw_uint32(br, &uval, bits)) {
        return false;
    }
    
    // Sign extend if necessary
    if (bits < 32 && (uval & (1u << (bits - 1)))) {
        uval |= (0xFFFFFFFF << bits);
    }
    
    *val = (int32_t)uval;
    return true;
}

static FLAC__bool bitreader_read_unary_unsigned(BitReader *br, uint32_t *val) {
    *val = 0;
    uint32_t bit;
    
    while (true) {
        if (!bitreader_read_raw_uint32(br, &bit, 1)) {
            return false;
        }
        
        if (bit == 0) {
            break;
        }
        
        (*val)++;
    }
    
    return true;
}

static FLAC__bool bitreader_read_rice_signed(BitReader *br, int32_t *val, uint32_t parameter) {
    uint32_t quotient;
    if (!bitreader_read_unary_unsigned(br, &quotient)) {
        return false;
    }
    
    uint32_t remainder = 0;
    if (parameter > 0) {
        if (!bitreader_read_raw_uint32(br, &remainder, parameter)) {
            return false;
        }
    }
    
    uint32_t uval = (quotient << parameter) | remainder;
    
    // Convert from unsigned to signed
    if (uval & 1) {
        *val = -((int32_t)(uval >> 1)) - 1;
    } else {
        *val = (int32_t)(uval >> 1);
    }
    
    return true;
}

// FLAC's read_residual_partitioned_rice_ function (adapted from stream_decoder.c)
static FLAC__bool read_residual_partitioned_rice_(
    BitReader *br,
    uint32_t predictor_order,
    uint32_t partition_order,
    FLAC__EntropyCodingMethod_PartitionedRiceContents *partitioned_rice_contents,
    FLAC__int32 *residual,
    uint32_t blocksize,
    FLAC__bool is_extended
) {
    FLAC__uint32 rice_parameter;
    int i;
    uint32_t partition, sample, u;
    const uint32_t partitions = 1u << partition_order;
    const uint32_t partition_samples = blocksize >> partition_order;
    const uint32_t plen = is_extended ? 
        FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE2_PARAMETER_LEN : 
        FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_PARAMETER_LEN;
    const uint32_t pesc = is_extended ? 
        FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE2_ESCAPE_PARAMETER : 
        FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_ESCAPE_PARAMETER;

    // Ensure we have enough space for parameters and raw_bits
    if (!partitioned_rice_contents->parameters) {
        partitioned_rice_contents->parameters = malloc(sizeof(uint32_t) * partitions);
        if (!partitioned_rice_contents->parameters) return false;
    }
    if (!partitioned_rice_contents->raw_bits) {
        partitioned_rice_contents->raw_bits = malloc(sizeof(uint32_t) * partitions);
        if (!partitioned_rice_contents->raw_bits) return false;
    }

    sample = 0;
    for (partition = 0; partition < partitions; partition++) {
        if (!bitreader_read_raw_uint32(br, &rice_parameter, plen))
            return false;
        
        partitioned_rice_contents->parameters[partition] = rice_parameter;
        
        if (rice_parameter < pesc) {
            // Rice coding
            partitioned_rice_contents->raw_bits[partition] = 0;
            u = (partition == 0) ? partition_samples - predictor_order : partition_samples;
            
            for (uint32_t j = 0; j < u; j++) {
                if (!bitreader_read_rice_signed(br, &residual[sample + j], rice_parameter)) {
                    return false;
                }
            }
            sample += u;
        } else {
            // Escape coding - read raw bits
            if (!bitreader_read_raw_uint32(br, &rice_parameter, FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_RAW_LEN))
                return false;
            
            partitioned_rice_contents->raw_bits[partition] = rice_parameter;
            
            if (rice_parameter == 0) {
                // All zeros
                for (u = (partition == 0) ? predictor_order : 0; u < partition_samples; u++, sample++) {
                    residual[sample] = 0;
                }
            } else {
                // Read raw values
                for (u = (partition == 0) ? predictor_order : 0; u < partition_samples; u++, sample++) {
                    if (!bitreader_read_raw_int32(br, &i, rice_parameter))
                        return false;
                    residual[sample] = i;
                }
            }
        }
    }

    return true;
}

// Main entropy decoding function
static FLAC__bool decode_entropy_coded_data(
    BitReader *br,
    FLAC__int32 *residual,
    uint32_t residual_samples,
    uint32_t predictor_order,
    FLAC__EntropyCodingMethod *ecm
) {
    uint32_t entropy_coding_method_type;
    uint32_t partition_order;
    
    // Read entropy coding method type (2 bits)
    if (!bitreader_read_raw_uint32(br, &entropy_coding_method_type, FLAC__ENTROPY_CODING_METHOD_TYPE_LEN))
        return false;
    
    // Read partition order (4 bits)
    if (!bitreader_read_raw_uint32(br, &partition_order, FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_ORDER_LEN))
        return false;
    
    ecm->type = entropy_coding_method_type;
    ecm->partitioned_rice.order = partition_order;
    
    const FLAC__bool is_extended = (entropy_coding_method_type == FLAC__ENTROPY_CODING_METHOD_TYPE_PARTITIONED_RICE2);
    
    // Calculate blocksize from residual_samples and predictor_order
    uint32_t blocksize = residual_samples + predictor_order;
    
    // Read the partitioned rice data
    FLAC__EntropyCodingMethod_PartitionedRiceContents *contents = 
        (FLAC__EntropyCodingMethod_PartitionedRiceContents*)ecm->partitioned_rice.contents;
    
    return read_residual_partitioned_rice_(
        br,
        predictor_order,
        partition_order,
        contents,
        residual,
        blocksize,
        is_extended
    );
}

int main(int argc, char *argv[]) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: %s input.bin num_samples [predictor_order] > output.bin\n", argv[0]);
        fprintf(stderr, "Decodes FLAC entropy-coded data back to residuals\n");
        fprintf(stderr, "  input.bin: entropy-coded bitstream file\n");
        fprintf(stderr, "  num_samples: number of residual samples to decode\n");
        fprintf(stderr, "  predictor_order: predictor order (optional, defaults to %d)\n", FLAC_DEFAULT_PREDICTOR_ORDER);
        return EXIT_FAILURE;
    }

    // Parse number of samples argument
    uint32_t num_samples = (uint32_t)strtoul(argv[2], NULL, 10);
    if (num_samples == 0) {
        fprintf(stderr, "Invalid number of samples: %s\n", argv[2]);
        return EXIT_FAILURE;
    }

    // Parse predictor order from command line or use default
    uint32_t predictor_order = FLAC_DEFAULT_PREDICTOR_ORDER;
    if (argc == 4) {
        char *endptr;
        long parsed_order = strtol(argv[3], &endptr, 10);
        if (*endptr != '\0' || parsed_order < 0 || parsed_order > 32) {
            fprintf(stderr, "Error: predictor_order must be an integer between 0 and 32\n");
            return EXIT_FAILURE;
        }
        predictor_order = (uint32_t)parsed_order;
    }

    // Open input file
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) {
        fprintf(stderr, "Error opening input file\n");
        return EXIT_FAILURE;
    }

    // Get file size
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    // Read entire file into buffer
    uint8_t *input_buffer = malloc((size_t)file_size);
    if (!input_buffer) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(fp);
        return EXIT_FAILURE;
    }

    if (fread(input_buffer, 1, (size_t)file_size, fp) != (size_t)file_size) {
        fprintf(stderr, "Error reading input file\n");
        free(input_buffer);
        fclose(fp);
        return EXIT_FAILURE;
    }
    fclose(fp);

    // Create BitReader
    BitReader *br = bitreader_new(input_buffer, (size_t)file_size);
    if (!br) {
        fprintf(stderr, "Failed to create BitReader\n");
        free(input_buffer);
        return EXIT_FAILURE;
    }

    // Set up decoder parameters (these should match the encoder)
    
    // Allocate residual buffer
    FLAC__int32 *residual = malloc(sizeof(FLAC__int32) * num_samples);
    if (!residual) {
        fprintf(stderr, "Memory allocation failed\n");
        bitreader_delete(br);
        free(input_buffer);
        return EXIT_FAILURE;
    }

    // Allocate entropy coding method structure
    FLAC__EntropyCodingMethod ecm;
    FLAC__EntropyCodingMethod_PartitionedRiceContents contents;
    contents.parameters = NULL;
    contents.raw_bits = NULL;
    ecm.partitioned_rice.contents = &contents;

    // Decode the entropy coded data
    if (!decode_entropy_coded_data(br, residual, num_samples, predictor_order, &ecm)) {
        fprintf(stderr, "Failed to decode entropy coded data\n");
        free(residual);
        free(contents.parameters);
        free(contents.raw_bits);
        bitreader_delete(br);
        free(input_buffer);
        return EXIT_FAILURE;
    }

    // Write decoded residuals to stdout
    fwrite(residual, sizeof(FLAC__int32), num_samples, stdout);

    // Print decoding statistics to stderr
    fprintf(stderr, "Decoded %u residual samples\n", num_samples);
    fprintf(stderr, "Entropy coding method: %s\n", 
            ecm.type == FLAC__ENTROPY_CODING_METHOD_TYPE_PARTITIONED_RICE2 ? "RICE2" : "RICE");
    uint32_t partitions = 1u << ecm.partitioned_rice.order;
    fprintf(stderr, "Partition order: %u (%u partitions)\n", 
            ecm.partitioned_rice.order, partitions);
    
    // Print partition information
    for (uint32_t i = 0; i < partitions; i++) {
        if (contents.raw_bits[i] > 0) {
            fprintf(stderr, "Partition %u: escape coding, %u raw bits\n", 
                    i, contents.raw_bits[i]);
        } else {
            fprintf(stderr, "Partition %u: Rice parameter %u\n", 
                    i, contents.parameters[i]);
        }
    }

    // Cleanup
    free(residual);
    free(contents.parameters);
    free(contents.raw_bits);
    bitreader_delete(br);
    free(input_buffer);

    return EXIT_SUCCESS;
} 