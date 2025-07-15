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
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }
    
    const char *input_filename = argv[1];
    const char *output_filename = argv[2];
    
    // Read binary input file
    FILE *input_file = fopen(input_filename, "rb");
    if (!input_file) {
        fprintf(stderr, "Error: Cannot open input file %s\n", input_filename);
        return 1;
    }
    
    // Get file size
    struct stat st;
    if (stat(input_filename, &st) != 0) {
        fprintf(stderr, "Error: Cannot stat input file\n");
        fclose(input_file);
        return 1;
    }
    
    if (st.st_size % 4 != 0) {
        fprintf(stderr, "Error: File size must be multiple of 4 bytes\n");
        fclose(input_file);
        return 1;
    }
    
    unsigned int num_samples = st.st_size / 4;
    
    // Warn if samples > 10,000
    if (num_samples > 10000) {
        fprintf(stderr, "Warning: Number of samples (%u) is larger than typical FLAC block size (max 10,000)\n", num_samples);
    }
    
    // Allocate buffer for samples
    FLAC__int32 *samples = (FLAC__int32 *)malloc(num_samples * sizeof(FLAC__int32));
    if (!samples) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(input_file);
        return 1;
    }
    
    // Read samples from file
    if (fread(samples, sizeof(FLAC__int32), num_samples, input_file) != num_samples) {
        fprintf(stderr, "Error: Failed to read all samples\n");
        free(samples);
        fclose(input_file);
        return 1;
    }
    fclose(input_file);
    
    // Open output file
    output_file = fopen(output_filename, "wb");
    if (!output_file) {
        fprintf(stderr, "Error: Cannot open output file %s\n", output_filename);
        free(samples);
        return 1;
    }
    
    // Create encoder
    FLAC__StreamEncoder *encoder = FLAC__stream_encoder_new();
    if (!encoder) {
        fprintf(stderr, "Error: Failed to create encoder\n");
        free(samples);
        fclose(output_file);
        return 1;
    }
    
    // Configure encoder for entropy coding
    FLAC__stream_encoder_set_channels(encoder, 1);  // Mono
    FLAC__stream_encoder_set_bits_per_sample(encoder, 32);
    FLAC__stream_encoder_set_sample_rate(encoder, 44100);
    FLAC__stream_encoder_set_blocksize(encoder, num_samples);  // Block size = number of samples
    FLAC__stream_encoder_set_max_lpc_order(encoder, 1);  // Enable LPC path (will be forced to 0 internally)
    FLAC__stream_encoder_set_compression_level(encoder, 0);  // Minimum compression
    
    // FORCE LPC PATH: Disable other subframe types
    fprintf(stderr, "Forcing LPC path by disabling other subframes...\n");
    FLAC__stream_encoder_disable_constant_subframes(encoder, true);
    FLAC__stream_encoder_disable_fixed_subframes(encoder, true);
    FLAC__stream_encoder_disable_verbatim_subframes(encoder, true);
    
    // Initialize encoder
    if (FLAC__stream_encoder_init_stream(encoder, write_callback, NULL, NULL, NULL, NULL) != FLAC__STREAM_ENCODER_INIT_STATUS_OK) {
        fprintf(stderr, "Error: Failed to initialize encoder\n");
        FLAC__stream_encoder_delete(encoder);
        free(samples);
        fclose(output_file);
        return 1;
    }
    
    // Encode samples
    const FLAC__int32 *buffer[1] = { samples };
    if (!FLAC__stream_encoder_process(encoder, buffer, num_samples)) {
        fprintf(stderr, "Error: Failed to encode samples\n");
        FLAC__stream_encoder_delete(encoder);
        free(samples);
        fclose(output_file);
        return 1;
    }
    
    // Finish encoding
    FLAC__stream_encoder_finish(encoder);
    
    // Clean up
    FLAC__stream_encoder_delete(encoder);
    free(samples);
    fclose(output_file);
    
    fprintf(stderr, "Successfully encoded %u samples\n", num_samples);
    return 0;
}
