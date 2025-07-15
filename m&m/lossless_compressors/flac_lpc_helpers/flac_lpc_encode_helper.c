#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <FLAC/stream_encoder.h>
#include <share/private.h>

static FILE *output_file = NULL;

static FLAC__StreamEncoderWriteStatus write_callback(const FLAC__StreamEncoder *encoder, const FLAC__byte buffer[], size_t bytes, unsigned samples, unsigned current_frame, void *client_data) {
    if (output_file) {
        fwrite(buffer, 1, bytes, output_file);
    }
    return FLAC__STREAM_ENCODER_WRITE_STATUS_OK;
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
    
    // Warn if too many samples
    if (num_samples > 10000) {
        fprintf(stderr, "Warning: Number of samples (%u) is larger than typical FLAC block size (max 10,000)\n", num_samples);
    }
    
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
    
    // Create temporary FLAC file
    char temp_flac_filename[] = "/tmp/flac_temp_XXXXXX.flac";
    int temp_fd = mkstemps(temp_flac_filename, 5);
    if (temp_fd == -1) {
        fprintf(stderr, "Error: Cannot create temporary file\n");
        free(samples);
        return 1;
    }
    close(temp_fd);
    
    // Open temporary output file
    output_file = fopen(temp_flac_filename, "wb");
    if (!output_file) {
        fprintf(stderr, "Error: Cannot open temporary output file %s\n", temp_flac_filename);
        free(samples);
        return 1;
    }
    
    // Create encoder
    FLAC__StreamEncoder *encoder = FLAC__stream_encoder_new();
    if (!encoder) {
        fprintf(stderr, "Error: Cannot create FLAC encoder\n");
        fclose(output_file);
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
        fclose(output_file);
        free(samples);
        return 1;
    }
    
    // Create buffer for mono channel
    const FLAC__int32 *buffer[1] = { samples };
    
    // Process all samples in one block
    if (!FLAC__stream_encoder_process(encoder, buffer, num_samples)) {
        fprintf(stderr, "Error: Failed to process samples\n");
        free(samples);
        FLAC__stream_encoder_delete(encoder);
        fclose(output_file);
        return 1;
    }
    
    // Finish encoding
    if (!FLAC__stream_encoder_finish(encoder)) {
        fprintf(stderr, "Error: Failed to finish encoding\n");
        free(samples);
        FLAC__stream_encoder_delete(encoder);
        fclose(output_file);
        return 1;
    }
    
    // Clean up encoder
    FLAC__stream_encoder_delete(encoder);
    fclose(output_file);
    
    // Now extract estimator bits from the temporary FLAC file
    FILE *temp_flac = fopen(temp_flac_filename, "rb");
    if (!temp_flac) {
        fprintf(stderr, "Error: Cannot open temporary FLAC file for reading\n");
        free(samples);
        return 1;
    }
    
    // Read FLAC file data
    fseek(temp_flac, 0, SEEK_END);
    long flac_size = ftell(temp_flac);
    fseek(temp_flac, 0, SEEK_SET);
    
    unsigned char *flac_data = malloc(flac_size);
    if (!flac_data) {
        fprintf(stderr, "Error: Cannot allocate memory for FLAC data\n");
        fclose(temp_flac);
        free(samples);
        return 1;
    }
    
    fread(flac_data, 1, flac_size, temp_flac);
    fclose(temp_flac);
    
    // Find magic markers
    const unsigned char begin_marker[] = {0x4D, 0x55, 0x47, 0x49}; // "MUGI"
    const unsigned char end_marker[] = {0x47, 0x4F, 0x44, 0x55};   // "GODU"
    
    long begin_pos = -1, end_pos = -1;
    
    for (long i = 0; i <= flac_size - 4; i++) {
        if (memcmp(flac_data + i, begin_marker, 4) == 0) {
            begin_pos = i;
        }
        if (memcmp(flac_data + i, end_marker, 4) == 0) {
            end_pos = i;
            // Don't break, continue searching to find the last occurrence
        }
    }
    
    if (begin_pos == -1) {
        fprintf(stderr, "Error: Begin magic marker not found\n");
        free(flac_data);
        free(samples);
        return 1;
    }
    
    // Extract estimator bits from BEGIN marker to end of file (if END marker not found)
    long estimator_start = begin_pos + 4;
    long estimator_length;
    
    if (end_pos == -1) {
        fprintf(stderr, "Warning: End magic marker not found, extracting from BEGIN to end of file\n");
        estimator_length = flac_size - estimator_start;
    } else {
        if (begin_pos >= end_pos) {
            fprintf(stderr, "Error: Magic markers in wrong order\n");
            free(flac_data);
            free(samples);
            return 1;
        }
        estimator_length = end_pos - estimator_start;
    }
    
    if (estimator_length <= 0) {
        fprintf(stderr, "Error: No estimator bits found\n");
        free(flac_data);
        free(samples);
        return 1;
    }
    
    // Extract estimator method from first byte
    if (estimator_length > 0) {
        unsigned char method_byte = flac_data[estimator_start];
        unsigned int estimator_method = (method_byte >> 6) & 0x3;
        fprintf(stderr, "Estimator Method Index: %u\n", estimator_method);
    }
    
    // Write estimator bits to output file
    FILE *output = fopen(output_filename, "wb");
    if (!output) {
        fprintf(stderr, "Error: Cannot open output file %s\n", output_filename);
        free(flac_data);
        free(samples);
        return 1;
    }
    
    fwrite(flac_data + estimator_start, 1, estimator_length, output);
    fclose(output);
    
    // Clean up
    free(flac_data);
    free(samples);
    unlink(temp_flac_filename);
    
    fprintf(stderr, "Successfully encoded %u samples and extracted %ld estimator bits\n", num_samples, estimator_length);
    
    return 0;
} 