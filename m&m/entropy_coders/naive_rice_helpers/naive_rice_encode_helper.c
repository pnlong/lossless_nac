#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* Constants matching Python implementation */
#define K_DEFAULT 1

/* Simple bit writer */
typedef struct {
    uint8_t *buffer;
    size_t capacity;
    size_t bytes;
    uint32_t bits;
    uint32_t total_bits;
} BitWriter;

/* Create new bit writer */
static BitWriter* bitwriter_new(size_t capacity) {
    BitWriter *bw = malloc(sizeof(BitWriter));
    if (!bw) return NULL;
    
    bw->buffer = malloc(capacity);
    if (!bw->buffer) {
        free(bw);
        return NULL;
    }
    
    bw->capacity = capacity;
    bw->bytes = 0;
    bw->bits = 0;
    bw->total_bits = 0;
    return bw;
}

/* Free bit writer */
static void bitwriter_delete(BitWriter *bw) {
    if (bw) {
        free(bw->buffer);
        free(bw);
    }
}

/* Write a single bit */
static int bitwriter_write_bit(BitWriter *bw, int bit) {
    if (bw->bytes >= bw->capacity - 4) return 0; /* Buffer full */
    
    if (bit) {
        bw->bits |= (1u << (31 - (bw->total_bits % 32)));
    }
    bw->total_bits++;
    
    /* Flush complete bytes */
    while (bw->total_bits >= 8) {
        bw->buffer[bw->bytes++] = (bw->bits >> 24) & 0xFF;
        bw->bits <<= 8;
        bw->total_bits -= 8;
    }
    
    return 1;
}

/* Write multiple bits */
static int bitwriter_write_bits(BitWriter *bw, uint32_t val, int bits) {
    for (int i = bits - 1; i >= 0; i--) {
        if (!bitwriter_write_bit(bw, (val >> i) & 1)) return 0;
    }
    return 1;
}

/* Flush remaining bits */
static void bitwriter_flush(BitWriter *bw) {
    if (bw->total_bits > 0) {
        bw->buffer[bw->bytes++] = (bw->bits >> 24) & 0xFF;
        bw->bits = 0;
        bw->total_bits = 0;
    }
}

/* Convert signed to unsigned (matching Python int_to_pos) */
static uint32_t int_to_pos(int32_t x) {
    if (x >= 0) {
        return 2 * x;
    } else {
        return -2 * x - 1;
    }
}

/* Naive Rice encode function (matching Python logic) */
static int naive_rice_encode(const int32_t *nums, size_t num_samples, int k, int is_nums_signed, BitWriter *bw) {
    for (size_t i = 0; i < num_samples; i++) {
        uint32_t x;
        
        /* Convert to unsigned if needed */
        if (is_nums_signed) {
            x = int_to_pos(nums[i]);
        } else {
            x = (uint32_t)nums[i];
        }
        
        /* Compute quotient and remainder */
        uint32_t q = x >> k;  /* quotient = x / 2^k */
        uint32_t r = x & ((1u << k) - 1);  /* remainder = x % 2^k */
        
        /* Encode quotient with unary coding (q ones followed by zero) */
        for (uint32_t j = 0; j < q; j++) {
            if (!bitwriter_write_bit(bw, 1)) return 0;
        }
        if (!bitwriter_write_bit(bw, 0)) return 0;
        
        /* Encode remainder with binary coding using k bits */
        if (!bitwriter_write_bits(bw, r, k)) return 0;
    }
    
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc != 2 && argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> [k_parameter]\n", argv[0]);
        fprintf(stderr, "  input_file: binary file containing 32-bit signed integers\n");
        fprintf(stderr, "  k_parameter: Rice parameter (default: %d)\n", K_DEFAULT);
        return 1;
    }
    
    const char *input_filename = argv[1];
    int k = (argc >= 3) ? atoi(argv[2]) : K_DEFAULT;
    
    if (k < 0 || k > 31) {
        fprintf(stderr, "Error: k parameter must be between 0 and 31\n");
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
    
    if (file_size <= 0 || file_size % 4 != 0) {
        fprintf(stderr, "Error: Invalid file size %ld (must be multiple of 4)\n", file_size);
        fclose(input_file);
        return 1;
    }
    
    size_t num_samples = file_size / 4;
    
    /* Read samples */
    int32_t *samples = malloc(file_size);
    if (!samples) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(input_file);
        return 1;
    }
    
    if (fread(samples, sizeof(int32_t), num_samples, input_file) != num_samples) {
        fprintf(stderr, "Error: Failed to read all samples\n");
        free(samples);
        fclose(input_file);
        return 1;
    }
    fclose(input_file);
    
    /* Create bit writer with generous capacity */
    size_t capacity = file_size * 10; /* Should be enough for Rice expansion */
    BitWriter *bw = bitwriter_new(capacity);
    if (!bw) {
        fprintf(stderr, "Error: Failed to create bit writer\n");
        free(samples);
        return 1;
    }
    
    /* Encode using naive Rice coding */
    if (!naive_rice_encode(samples, num_samples, k, 1, bw)) {
        fprintf(stderr, "Error: Rice encoding failed\n");
        bitwriter_delete(bw);
        free(samples);
        return 1;
    }
    
    /* Flush any remaining bits */
    bitwriter_flush(bw);
    
    /* Output encoded data to stdout */
    fwrite(bw->buffer, 1, bw->bytes, stdout);
    
    /* Cleanup */
    bitwriter_delete(bw);
    free(samples);
    
    return 0;
} 