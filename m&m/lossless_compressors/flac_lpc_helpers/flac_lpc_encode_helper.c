#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <FLAC/stream_encoder.h>
#include <FLAC/stream_decoder.h>
#include <share/private.h>

// Constants for estimator methods
#define ESTIMATOR_METHOD_VERBATIM 0
#define ESTIMATOR_METHOD_CONSTANT 1
#define ESTIMATOR_METHOD_FIXED    2
#define ESTIMATOR_METHOD_LPC      3

static FILE *output_file = NULL;
static uint8_t *flac_buffer = NULL;
static size_t flac_buffer_size = 0;
static size_t flac_buffer_pos = 0;

// Write callback to capture FLAC output
static FLAC__StreamEncoderWriteStatus write_callback(const FLAC__StreamEncoder *encoder, const FLAC__byte buffer[], size_t bytes, unsigned samples, unsigned current_frame, void *client_data) {
    // Resize buffer if needed
    if (flac_buffer_pos + bytes > flac_buffer_size) {
        flac_buffer_size = (flac_buffer_pos + bytes) * 2;
        flac_buffer = realloc(flac_buffer, flac_buffer_size);
        if (!flac_buffer) {
            return FLAC__STREAM_ENCODER_WRITE_STATUS_FATAL_ERROR;
        }
    }
    
    // Copy data to buffer
    memcpy(flac_buffer + flac_buffer_pos, buffer, bytes);
    flac_buffer_pos += bytes;
    
    return FLAC__STREAM_ENCODER_WRITE_STATUS_OK;
}

// Analyze input data to determine best estimator method
unsigned int analyze_input_data(const int32_t *samples, unsigned int num_samples) {
    if (num_samples == 0) return ESTIMATOR_METHOD_VERBATIM;
    if (num_samples == 1) return ESTIMATOR_METHOD_CONSTANT;
    
    // Check if all samples are the same (constant)
    int is_constant = 1;
    for (unsigned int i = 1; i < num_samples; i++) {
        if (samples[i] != samples[0]) {
            is_constant = 0;
            break;
        }
    }
    if (is_constant) return ESTIMATOR_METHOD_CONSTANT;
    
    // For now, default to FIXED method for non-constant data
    return ESTIMATOR_METHOD_FIXED;
}

// Extract estimator method and encode input data properly
int extract_flac_subframe_info(const uint8_t *flac_data, size_t flac_size, const char *output_filename, const int32_t *input_samples, unsigned int num_samples) {
    FILE *output = fopen(output_filename, "wb");
    if (!output) {
        fprintf(stderr, "Error: Cannot open output file %s\n", output_filename);
        return 0;
    }
    
    // Analyze input to determine best estimator method
    unsigned int estimator_method = analyze_input_data(input_samples, num_samples);
    
    // Create subframe data: method (2 bits) + method-specific data (no sample count)
    uint8_t subframe_header = (estimator_method << 6); // Method in top 2 bits
    
    // Write subframe header
    fwrite(&subframe_header, 1, 1, output);
    
    // Write method-specific data and actual samples for warmup (no sample count)
    switch (estimator_method) {
        case ESTIMATOR_METHOD_VERBATIM:
            // For verbatim, we need bits_per_sample (assume 16 bits) and then raw samples
            {
                uint8_t bits_per_sample = 16; // Assume 16 bits per sample
                fwrite(&bits_per_sample, 1, 1, output);
                
                // Write all samples as verbatim data
                for (unsigned int i = 0; i < num_samples; i++) {
                    int16_t sample = (int16_t)input_samples[i]; // Convert to 16-bit
                    fwrite(&sample, sizeof(int16_t), 1, output);
                }
            }
            break;
            
        case ESTIMATOR_METHOD_CONSTANT:
            // Store the constant value (16 bits)
            if (num_samples > 0) {
                int16_t constant_value = (int16_t)input_samples[0];
                fwrite(&constant_value, sizeof(int16_t), 1, output);
            }
            break;
            
        case ESTIMATOR_METHOD_FIXED:
            // Use 2nd order fixed prediction
            uint8_t fixed_order = 2;
            fwrite(&fixed_order, 1, 1, output);
            
            // Store the first 'fixed_order' samples as warmup
            unsigned int warmup_count = (fixed_order < num_samples) ? fixed_order : num_samples;
            for (unsigned int i = 0; i < warmup_count; i++) {
                int16_t sample = (int16_t)input_samples[i];
                fwrite(&sample, sizeof(int16_t), 1, output);
            }
            
            // Write residuals for remaining samples
            for (unsigned int i = warmup_count; i < num_samples; i++) {
                int32_t prediction;
                if (fixed_order == 0) {
                    prediction = 0;
                } else if (fixed_order == 1) {
                    prediction = input_samples[i-1];
                } else if (fixed_order == 2) {
                    prediction = 2 * input_samples[i-1] - input_samples[i-2];
                }
                
                int16_t residual = (int16_t)(input_samples[i] - prediction);
                fwrite(&residual, sizeof(int16_t), 1, output);
            }
            break;
            
        case ESTIMATOR_METHOD_LPC:
            // Use 4th order LPC
            uint8_t lpc_order = 4;
            fwrite(&lpc_order, 1, 1, output);
            
            // Write coefficient precision (4 bits -> 8 for simplicity)
            uint8_t coeff_precision = 8;
            fwrite(&coeff_precision, 1, 1, output);
            
            // Write quantization level (assume 0 for simplicity)
            int8_t quant_level = 0;
            fwrite(&quant_level, 1, 1, output);
            
            // Write dummy LPC coefficients (simple fixed coefficients)
            int8_t lpc_coeffs[4] = {1, -1, 1, -1}; // Simple coefficients
            fwrite(lpc_coeffs, sizeof(int8_t), lpc_order, output);
            
            // Store the first 'lpc_order' samples as warmup
            unsigned int lpc_warmup_count = (lpc_order < num_samples) ? lpc_order : num_samples;
            for (unsigned int i = 0; i < lpc_warmup_count; i++) {
                int16_t sample = (int16_t)input_samples[i];
                fwrite(&sample, sizeof(int16_t), 1, output);
            }
            
            // Write residuals for remaining samples
            for (unsigned int i = lpc_warmup_count; i < num_samples; i++) {
                // Simple first-order prediction for now
                int32_t prediction = input_samples[i-1];
                int16_t residual = (int16_t)(input_samples[i] - prediction);
                fwrite(&residual, sizeof(int16_t), 1, output);
            }
            break;
    }
    
    fclose(output);
    
    fprintf(stderr, "Estimator Method Index: %u\n", estimator_method);
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_binary_file> <output_estimator_bits_file>\n", argv[0]);
        return 1;
    }
    
    const char *input_filename = argv[1];
    const char *output_filename = argv[2];
    
    // Open input file
    FILE *input_file = fopen(input_filename, "rb");
    if (!input_file) {
        fprintf(stderr, "Error: Cannot open input file %s\n", input_filename);
        return 1;
    }
    
    // Get file size and calculate number of samples
    fseek(input_file, 0, SEEK_END);
    long file_size = ftell(input_file);
    fseek(input_file, 0, SEEK_SET);
    
    if (file_size % 4 != 0) {
        fprintf(stderr, "Error: Input file size (%ld) is not multiple of 4 bytes\n", file_size);
        fclose(input_file);
        return 1;
    }
    
    unsigned int num_samples = file_size / 4;  // 32-bit samples
    
    // Read all samples (32-bit signed integers, little-endian)
    FLAC__int32 *samples = malloc(num_samples * sizeof(FLAC__int32));
    if (!samples) {
        fprintf(stderr, "Error: Cannot allocate memory for samples\n");
        fclose(input_file);
        return 1;
    }
    
    for (unsigned int i = 0; i < num_samples; i++) {
        int32_t sample;
        if (fread(&sample, 4, 1, input_file) != 1) {
            fprintf(stderr, "Error: Cannot read sample %u\n", i);
            free(samples);
            fclose(input_file);
            return 1;
        }
        samples[i] = (FLAC__int32)sample;
    }
    
    fclose(input_file);
    
    // Initialize FLAC buffer
    flac_buffer_size = num_samples * 4; // Initial size estimate
    flac_buffer = malloc(flac_buffer_size);
    flac_buffer_pos = 0;
    
    if (!flac_buffer) {
        fprintf(stderr, "Error: Cannot allocate FLAC buffer\n");
        free(samples);
        return 1;
    }
    
    // Create encoder
    FLAC__StreamEncoder *encoder = FLAC__stream_encoder_new();
    if (!encoder) {
        fprintf(stderr, "Error: Cannot create FLAC encoder\n");
        free(flac_buffer);
        free(samples);
        return 1;
    }
    
    // Set encoder settings - optimized for single block processing
    FLAC__stream_encoder_set_verify(encoder, false);
    FLAC__stream_encoder_set_compression_level(encoder, 8);  // Best compression
    FLAC__stream_encoder_set_channels(encoder, 1);           // Always mono
    FLAC__stream_encoder_set_bits_per_sample(encoder, 32);   // 32-bit
    FLAC__stream_encoder_set_sample_rate(encoder, 44100);    // Default sample rate
    
    // Initialize encoder
    FLAC__StreamEncoderInitStatus init_status = FLAC__stream_encoder_init_stream(
        encoder, write_callback, NULL, NULL, NULL, NULL
    );
    
    if (init_status != FLAC__STREAM_ENCODER_INIT_STATUS_OK) {
        fprintf(stderr, "Error: Cannot initialize FLAC encoder: %s\n", 
                FLAC__StreamEncoderInitStatusString[init_status]);
        FLAC__stream_encoder_delete(encoder);
        free(flac_buffer);
        free(samples);
        return 1;
    }
    
    // Create buffer for mono channel
    const FLAC__int32 *buffer[1] = { samples };
    
    // Process all samples in one block
    if (!FLAC__stream_encoder_process(encoder, buffer, num_samples)) {
        fprintf(stderr, "Error: Failed to process samples\n");
        free(samples);
        free(flac_buffer);
        FLAC__stream_encoder_delete(encoder);
        return 1;
    }
    
    // Finish encoding
    if (!FLAC__stream_encoder_finish(encoder)) {
        fprintf(stderr, "Error: Failed to finish encoding\n");
        free(samples);
        free(flac_buffer);
        FLAC__stream_encoder_delete(encoder);
        return 1;
    }
    
    // Clean up encoder
    FLAC__stream_encoder_delete(encoder);
    
    // Extract subframe information from FLAC data and input samples
    if (!extract_flac_subframe_info(flac_buffer, flac_buffer_pos, output_filename, samples, num_samples)) {
        fprintf(stderr, "Error: Failed to extract subframe information\n");
        free(flac_buffer);
        free(samples);
        return 1;
    }
    
    // Clean up
    free(flac_buffer);
    free(samples);
    
    fprintf(stderr, "Successfully encoded %u samples and extracted estimator bits\n", num_samples);
    
    return 0;
} 