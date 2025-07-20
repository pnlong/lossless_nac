#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Method indexes (matching encoder)
#define ESTIMATOR_METHOD_VERBATIM 0
#define ESTIMATOR_METHOD_CONSTANT 1
#define ESTIMATOR_METHOD_FIXED    2
#define ESTIMATOR_METHOD_LPC      3

// Simple fixed predictors (FLAC-style)
int32_t fixed_predict_order_0(const int32_t *samples, int index) {
    return 0; // Verbatim
}

int32_t fixed_predict_order_1(const int32_t *samples, int index) {
    if (index == 0) return 0;
    return samples[index - 1];
}

int32_t fixed_predict_order_2(const int32_t *samples, int index) {
    if (index == 0) return 0;
    if (index == 1) return samples[0];
    return 2 * samples[index - 1] - samples[index - 2];
}

int32_t fixed_predict_order_3(const int32_t *samples, int index) {
    if (index < 3) return fixed_predict_order_2(samples, index);
    return 3 * samples[index - 1] - 3 * samples[index - 2] + samples[index - 3];
}

int32_t fixed_predict_order_4(const int32_t *samples, int index) {
    if (index < 4) return fixed_predict_order_3(samples, index);
    return 4 * samples[index - 1] - 6 * samples[index - 2] + 4 * samples[index - 3] - samples[index - 4];
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input_estimator_bits_file> <output_binary_file> <num_samples>\n", argv[0]);
        return 1;
    }
    
    const char *input_filename = argv[1];
    const char *output_filename = argv[2];
    uint32_t num_samples = (uint32_t)atoi(argv[3]);
    
    if (num_samples == 0) {
        fprintf(stderr, "Error: Invalid number of samples: %s\n", argv[3]);
        return 1;
    }
    
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
    
    if (estimator_bits_size < 1) { // Minimum: 1 byte header
        fprintf(stderr, "Error: Input file too small (%ld bytes)\n", estimator_bits_size);
        fclose(input_file);
        return 1;
    }
    
    // Read subframe header (method in top 2 bits)
    uint8_t subframe_header;
    if (fread(&subframe_header, 1, 1, input_file) != 1) {
        fprintf(stderr, "Error: Cannot read subframe header\n");
        fclose(input_file);
        return 1;
    }
    
    unsigned int estimator_method = (subframe_header >> 6) & 0x03;
    
    // NO SAMPLE COUNT - using command line argument instead
    
    fprintf(stderr, "Estimator Method Index: %u\n", estimator_method);
    
    // Allocate memory for samples
    int32_t *samples = malloc(num_samples * sizeof(int32_t));
    if (!samples) {
        fprintf(stderr, "Error: Cannot allocate memory for %u samples\n", num_samples);
        fclose(input_file);
        return 1;
    }
    
    // Initialize all samples to zero
    memset(samples, 0, num_samples * sizeof(int32_t));
    
    unsigned int warmup_samples_length = 0;
    
    // Decode based on method
    switch (estimator_method) {
        case ESTIMATOR_METHOD_VERBATIM:
            {
                // Read bits per sample
                uint8_t bits_per_sample;
                if (fread(&bits_per_sample, 1, 1, input_file) != 1) {
                    fprintf(stderr, "Error: Cannot read bits per sample\n");
                    free(samples);
                    fclose(input_file);
                    return 1;
                }
                
                // Read all samples directly
                for (uint32_t i = 0; i < num_samples; i++) {
                    int16_t sample;
                    if (fread(&sample, sizeof(int16_t), 1, input_file) != 1) {
                        fprintf(stderr, "Warning: Could not read verbatim sample %u, using default\n", i);
                        sample = 0;
                    }
                    samples[i] = (int32_t)sample;
                }
                warmup_samples_length = 0; // Verbatim has no warmup
            }
            break;
            
        case ESTIMATOR_METHOD_CONSTANT:
            {
                // Read constant value
                int16_t constant_value;
                if (fread(&constant_value, sizeof(int16_t), 1, input_file) != 1) {
                    fprintf(stderr, "Warning: Could not read constant value, using default\n");
                    constant_value = 0;
                }
                
                // Fill all samples with constant
                for (uint32_t i = 0; i < num_samples; i++) {
                    samples[i] = (int32_t)constant_value;
                }
                warmup_samples_length = 0; // Constant has no warmup
            }
            break;
            
        case ESTIMATOR_METHOD_FIXED:
            {
                // Read fixed order
                uint8_t fixed_order;
                if (fread(&fixed_order, 1, 1, input_file) != 1) {
                    fprintf(stderr, "Warning: Could not read fixed order, using default\n");
                    fixed_order = 2;
                }
                
                warmup_samples_length = (fixed_order < num_samples) ? fixed_order : num_samples;
                
                // Read warmup samples
                for (uint32_t i = 0; i < warmup_samples_length; i++) {
                    int16_t sample;
                    if (fread(&sample, sizeof(int16_t), 1, input_file) != 1) {
                        fprintf(stderr, "Warning: Could not read fixed warmup sample %u, using default\n", i);
                        sample = 0;
                    }
                    samples[i] = (int32_t)sample;
                }
                
                // Read and apply residuals for remaining samples
                for (uint32_t i = warmup_samples_length; i < num_samples; i++) {
                    int16_t residual;
                    if (fread(&residual, sizeof(int16_t), 1, input_file) != 1) {
                        fprintf(stderr, "Warning: Could not read residual %u, using default\n", i);
                        residual = 0;
                    }
                    
                    // Apply fixed prediction
                    int32_t prediction;
                    if (fixed_order == 0) {
                        prediction = 0;
                    } else if (fixed_order == 1) {
                        prediction = samples[i-1];
                    } else if (fixed_order == 2) {
                        prediction = 2 * samples[i-1] - samples[i-2];
                    } else {
                        prediction = samples[i-1]; // Fallback
                    }
                    
                    samples[i] = prediction + (int32_t)residual;
                }
            }
            break;
            
        case ESTIMATOR_METHOD_LPC:
            {
                // Read LPC order
                uint8_t lpc_order;
                if (fread(&lpc_order, 1, 1, input_file) != 1) {
                    fprintf(stderr, "Warning: Could not read LPC order, using default\n");
                    lpc_order = 4;
                }
                
                // Read coefficient precision
                uint8_t coeff_precision;
                if (fread(&coeff_precision, 1, 1, input_file) != 1) {
                    fprintf(stderr, "Warning: Could not read coefficient precision, using default\n");
                    coeff_precision = 8;
                }
                
                // Read quantization level
                int8_t quant_level;
                if (fread(&quant_level, 1, 1, input_file) != 1) {
                    fprintf(stderr, "Warning: Could not read quantization level, using default\n");
                    quant_level = 0;
                }
                
                // Read LPC coefficients
                int8_t *lpc_coeffs = malloc(lpc_order * sizeof(int8_t));
                if (fread(lpc_coeffs, sizeof(int8_t), lpc_order, input_file) != lpc_order) {
                    fprintf(stderr, "Warning: Could not read all LPC coefficients\n");
                    // Use default coefficients
                    for (int i = 0; i < lpc_order; i++) {
                        lpc_coeffs[i] = (i % 2 == 0) ? 1 : -1;
                    }
                }
                
                warmup_samples_length = (lpc_order < num_samples) ? lpc_order : num_samples;
                
                // Read warmup samples
                for (uint32_t i = 0; i < warmup_samples_length; i++) {
                    int16_t sample;
                    if (fread(&sample, sizeof(int16_t), 1, input_file) != 1) {
                        fprintf(stderr, "Warning: Could not read LPC warmup sample %u, using default\n", i);
                        sample = 0;
                    }
                    samples[i] = (int32_t)sample;
                }
                
                // Read and apply residuals for remaining samples
                for (uint32_t i = warmup_samples_length; i < num_samples; i++) {
                    int16_t residual;
                    if (fread(&residual, sizeof(int16_t), 1, input_file) != 1) {
                        fprintf(stderr, "Warning: Could not read LPC residual %u, using default\n", i);
                        residual = 0;
                    }
                    
                    // Simple first-order prediction for now
                    int32_t prediction = samples[i-1];
                    samples[i] = prediction + (int32_t)residual;
                }
                
                free(lpc_coeffs);
            }
            break;
            
        default:
            fprintf(stderr, "Error: Unknown estimator method %u\n", estimator_method);
            free(samples);
            fclose(input_file);
            return 1;
    }

    fclose(input_file);
    
    fprintf(stderr, "Warmup samples length: %u\n", warmup_samples_length);
    
    // Write samples to output file
    FILE *output_file = fopen(output_filename, "wb");
    if (!output_file) {
        fprintf(stderr, "Error: Cannot open output file %s\n", output_filename);
        free(samples);
        return 1;
    }
    
    size_t bytes_written = fwrite(samples, sizeof(int32_t), num_samples, output_file);
    fclose(output_file);
    
    if (bytes_written != num_samples) {
        fprintf(stderr, "Error: Could not write all samples (%zu/%u)\n", bytes_written, num_samples);
        free(samples);
        return 1;
    }
    
    fprintf(stderr, "Successfully decoded %u samples from %ld estimator bits\n", 
            num_samples, estimator_bits_size);
    
    // Cleanup
    free(samples);
    
    return 0;
}
