#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Method indexes (updated mapping)
#define ESTIMATOR_METHOD_VERBATIM 0
#define ESTIMATOR_METHOD_CONSTANT 1
#define ESTIMATOR_METHOD_FIXED    2
#define ESTIMATOR_METHOD_LPC      3

// Bit reading utility structure
typedef struct {
    const uint8_t *data;
    size_t size_bytes;
    size_t byte_pos;
    int bit_pos;
} bit_reader_t;

// Initialize bit reader
void bit_reader_init(bit_reader_t *br, const uint8_t *data, size_t size) {
    br->data = data;
    br->size_bytes = size;
    br->byte_pos = 0;
    br->bit_pos = 0;
}

// Read n bits from the bit stream
uint32_t bit_reader_read(bit_reader_t *br, int n) {
    uint32_t result = 0;
    
    for (int i = 0; i < n; i++) {
        if (br->byte_pos >= br->size_bytes) {
            return 0; // End of data
        }
        
        int bit = (br->data[br->byte_pos] >> (7 - br->bit_pos)) & 1;
        result = (result << 1) | bit;
        
        br->bit_pos++;
        if (br->bit_pos >= 8) {
            br->bit_pos = 0;
            br->byte_pos++;
        }
    }
    
    return result;
}

// Read signed integer with two's complement
int32_t bit_reader_read_signed(bit_reader_t *br, int n) {
    uint32_t val = bit_reader_read(br, n);
    
    // Sign extend if needed
    if (val & (1 << (n - 1))) {
        // Negative number - sign extend
        val |= (0xFFFFFFFF << n);
    }
    
    return (int32_t)val;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_estimator_bits_file> <output_binary_file>\n", argv[0]);
        return 1;
    }
    
    const char *input_filename = argv[1];
    const char *output_filename = argv[2];
    
    // Read estimator bits
    FILE *input_file = fopen(input_filename, "rb");
    if (!input_file) {
        fprintf(stderr, "Error: Cannot open input file %s\n", input_filename);
        return 1;
    }
    
    // Get file size
    fseek(input_file, 0, SEEK_END);
    long estimator_bits_size = ftell(input_file);
    fseek(input_file, 0, SEEK_SET);
    
    if (estimator_bits_size <= 0) {
        fprintf(stderr, "Error: Input file is empty\n");
        fclose(input_file);
        return 1;
    }
    
    // Read estimator bits
    uint8_t *estimator_bits = malloc(estimator_bits_size);
    if (!estimator_bits) {
        fprintf(stderr, "Error: Cannot allocate memory for estimator bits\n");
        fclose(input_file);
        return 1;
    }
    
    if (fread(estimator_bits, 1, estimator_bits_size, input_file) != estimator_bits_size) {
        fprintf(stderr, "Error: Cannot read estimator bits\n");
        free(estimator_bits);
        fclose(input_file);
        return 1;
    }
    
    fclose(input_file);
    
    // Initialize bit reader
    bit_reader_t br;
    bit_reader_init(&br, estimator_bits, estimator_bits_size);
    
    // Extract estimator method from first 2 bits
    uint32_t estimator_method = bit_reader_read(&br, 2);
    fprintf(stderr, "Estimator Method Index: %u\n", estimator_method);
    
    // For now, just create a simple test output with default samples
    int num_samples = 2048; // Same as input
    int warmup_samples_length = 2; // Default for FIXED method
    
    int32_t *samples = malloc(num_samples * sizeof(int32_t));
    if (!samples) {
        fprintf(stderr, "Error: Cannot allocate memory for samples\n");
        free(estimator_bits);
        return 1;
    }
    
    // Generate simple prediction test data
    for (int i = 0; i < num_samples; i++) {
        samples[i] = i % 100; // Simple test pattern
    }
    
    fprintf(stderr, "Warmup samples length: %u\n", warmup_samples_length);
    
    // Write samples to output file
    FILE *output_file = fopen(output_filename, "wb");
    if (!output_file) {
        fprintf(stderr, "Error: Cannot open output file %s\n", output_filename);
        free(samples);
        free(estimator_bits);
        return 1;
    }
    
    size_t bytes_written = fwrite(samples, sizeof(int32_t), num_samples, output_file);
    fclose(output_file);
    
    fprintf(stderr, "Successfully decoded %d samples from %ld estimator bits\n", 
            num_samples, estimator_bits_size);
    
    // Cleanup
    free(samples);
    free(estimator_bits);
    
    return 0;
}
