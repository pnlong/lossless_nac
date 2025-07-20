#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* Constants matching Python implementation */
#define K_DEFAULT 1

/* Simple bit reader */
typedef struct {
    const uint8_t *data;
    size_t size;
    size_t byte_pos;
    int bit_pos;
} BitReader;

/* Initialize bit reader */
static void bitreader_init(BitReader *br, const uint8_t *data, size_t size) {
    br->data = data;
    br->size = size;
    br->byte_pos = 0;
    br->bit_pos = 0;
}

/* Read a single bit */
static int bitreader_read_bit(BitReader *br) {
    if (br->byte_pos >= br->size) {
        return -1; /* End of stream */
    }
    
    uint8_t bit = (br->data[br->byte_pos] >> (7 - br->bit_pos)) & 1;
    
    br->bit_pos++;
    if (br->bit_pos >= 8) {
        br->bit_pos = 0;
        br->byte_pos++;
    }
    
    return bit;
}

/* Read multiple bits */
static uint32_t bitreader_read_bits(BitReader *br, int n) {
    uint32_t result = 0;
    
    for (int i = 0; i < n; i++) {
        int bit = bitreader_read_bit(br);
        if (bit < 0) return 0; /* End of stream */
        result = (result << 1) | bit;
    }
    
    return result;
}

/* Convert unsigned to signed (matching Python inverse_int_to_pos) */
static int32_t inverse_int_to_pos(uint32_t x) {
    if (x % 2 == 0) {
        return (int32_t)(x / 2);
    } else {
        return -(int32_t)((x + 1) / 2);
    }
}

/* Naive Rice decode function (matching Python logic) */
static int naive_rice_decode(BitReader *br, int32_t *nums, size_t num_samples, int k, int is_nums_signed) {
    for (size_t i = 0; i < num_samples; i++) {
        /* Read unary-coded quotient (count ones until zero) */
        uint32_t q = 0;
        int bit;
        while ((bit = bitreader_read_bit(br)) == 1) {
            q++;
            if (q > 1000000) { /* Safety check for corrupted data */
                fprintf(stderr, "Error: Quotient too large, possibly corrupted data\n");
                return 0;
            }
        }
        
        if (bit < 0) {
            fprintf(stderr, "Error: Unexpected end of stream while reading quotient\n");
            return 0;
        }
        
        /* Read k-bit remainder */
        uint32_t r = bitreader_read_bits(br, k);
        
        /* Reconstruct original number */
        uint32_t x = (q << k) | r;
        
        /* Convert back to signed if needed */
        if (is_nums_signed) {
            nums[i] = inverse_int_to_pos(x);
        } else {
            nums[i] = (int32_t)x;
        }
    }
    
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "Usage: %s <input_file> <num_samples> [k_parameter]\n", argv[0]);
        fprintf(stderr, "  input_file: binary file containing encoded Rice data\n");
        fprintf(stderr, "  num_samples: number of samples to decode\n");
        fprintf(stderr, "  k_parameter: Rice parameter (default: %d)\n", K_DEFAULT);
        return 1;
    }
    
    const char *input_filename = argv[1];
    size_t num_samples = (size_t)atol(argv[2]);
    int k = (argc >= 4) ? atoi(argv[3]) : K_DEFAULT;
    
    if (k < 0 || k > 31) {
        fprintf(stderr, "Error: k parameter must be between 0 and 31\n");
        return 1;
    }
    
    if (num_samples == 0) {
        fprintf(stderr, "Error: num_samples must be greater than 0\n");
        return 1;
    }
    
    /* Open input file */
    FILE *input_file = fopen(input_filename, "rb");
    if (!input_file) {
        fprintf(stderr, "Error: Cannot open input file %s\n", input_filename);
        return 1;
    }
    
    /* Get file size */
    fseek(input_file, 0, SEEK_END);
    long file_size = ftell(input_file);
    fseek(input_file, 0, SEEK_SET);
    
    if (file_size <= 0) {
        fprintf(stderr, "Error: Invalid file size %ld\n", file_size);
        fclose(input_file);
        return 1;
    }
    
    /* Read encoded data */
    uint8_t *encoded_data = malloc(file_size);
    if (!encoded_data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(input_file);
        return 1;
    }
    
    if (fread(encoded_data, 1, file_size, input_file) != (size_t)file_size) {
        fprintf(stderr, "Error: Failed to read encoded data\n");
        free(encoded_data);
        fclose(input_file);
        return 1;
    }
    fclose(input_file);
    
    /* Allocate output buffer */
    int32_t *samples = malloc(num_samples * sizeof(int32_t));
    if (!samples) {
        fprintf(stderr, "Error: Memory allocation failed for samples\n");
        free(encoded_data);
        return 1;
    }
    
    /* Initialize bit reader */
    BitReader br;
    bitreader_init(&br, encoded_data, file_size);
    
    /* Decode using naive Rice coding */
    if (!naive_rice_decode(&br, samples, num_samples, k, 1)) {
        fprintf(stderr, "Error: Rice decoding failed\n");
        free(samples);
        free(encoded_data);
        return 1;
    }
    
    /* Output decoded samples to stdout */
    fwrite(samples, sizeof(int32_t), num_samples, stdout);
    
    /* Cleanup */
    free(samples);
    free(encoded_data);
    
    return 0;
} 