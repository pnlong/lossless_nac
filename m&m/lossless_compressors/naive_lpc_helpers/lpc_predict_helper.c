#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* LPC prediction function matching Python naive_lpc logic */
int lpc_predict_samples(
    const float *lpc_coefficients,
    int order,
    const float *warmup_samples,
    int n_predicted_samples,
    float *predicted_samples
) {
    /* Initialize predicted samples array to zero */
    memset(predicted_samples, 0, n_predicted_samples * sizeof(float));
    
    /* Predict each sample iteratively */
    for (int i = 0; i < n_predicted_samples; i++) {
        float prediction = 0.0f;
        
        if (i < order) {
            /* Use some warmup samples and some predicted samples */
            /* previous_samples = warmup_samples[-(order-i):] + predicted_samples[:i] */
            /* This creates an array of length 'order' */
            
            /* First part: warmup_samples[-(order-i):] which is warmup_samples[i:order] */
            int num_warmup = order - i;
            for (int j = 0; j < num_warmup; j++) {
                int warmup_idx = i + j;
                int coeff_idx = order - 1 - j; /* Reverse order for dot product */
                prediction += lpc_coefficients[coeff_idx] * warmup_samples[warmup_idx];
            }
            
            /* Second part: predicted_samples[:i] */
            for (int j = 0; j < i; j++) {
                int pred_idx = j;
                int coeff_idx = order - 1 - num_warmup - j; /* Continue reverse order */
                prediction += lpc_coefficients[coeff_idx] * predicted_samples[pred_idx];
            }
        } else {
            /* Use only previous predicted samples */
            /* previous_samples = predicted_samples[(i-order):i] */
            for (int j = 0; j < order; j++) {
                int sample_idx = (i - order) + j;
                int coeff_idx = order - 1 - j; /* Reverse order for dot product */
                prediction += lpc_coefficients[coeff_idx] * predicted_samples[sample_idx];
            }
        }
        
        predicted_samples[i] = prediction;
    }
    
    return 0; /* Success */
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <coefficients_file> <warmup_file> <n_predicted_samples>\n", argv[0]);
        fprintf(stderr, "  coefficients_file: binary file containing float32 LPC coefficients\n");
        fprintf(stderr, "  warmup_file: binary file containing float32 warmup samples\n");
        fprintf(stderr, "  n_predicted_samples: number of samples to predict\n");
        return 1;
    }
    
    const char *coefficients_filename = argv[1];
    const char *warmup_filename = argv[2];
    int n_predicted_samples = atoi(argv[3]);
    
    if (n_predicted_samples <= 0) {
        fprintf(stderr, "Error: n_predicted_samples must be positive\n");
        return 1;
    }
    
    /* Read LPC coefficients */
    FILE *coeff_file = fopen(coefficients_filename, "rb");
    if (!coeff_file) {
        fprintf(stderr, "Error: Cannot open coefficients file %s\n", coefficients_filename);
        return 1;
    }
    
    /* Get coefficients file size to determine order */
    fseek(coeff_file, 0, SEEK_END);
    long coeff_file_size = ftell(coeff_file);
    fseek(coeff_file, 0, SEEK_SET);
    
    if (coeff_file_size <= 0 || coeff_file_size % sizeof(float) != 0) {
        fprintf(stderr, "Error: Invalid coefficients file size %ld\n", coeff_file_size);
        fclose(coeff_file);
        return 1;
    }
    
    int order = coeff_file_size / sizeof(float);
    
    /* Read coefficients */
    float *lpc_coefficients = malloc(order * sizeof(float));
    if (!lpc_coefficients) {
        fprintf(stderr, "Error: Memory allocation failed for coefficients\n");
        fclose(coeff_file);
        return 1;
    }
    
    if (fread(lpc_coefficients, sizeof(float), order, coeff_file) != (size_t)order) {
        fprintf(stderr, "Error: Failed to read coefficients\n");
        free(lpc_coefficients);
        fclose(coeff_file);
        return 1;
    }
    fclose(coeff_file);
    
    /* Read warmup samples */
    FILE *warmup_file = fopen(warmup_filename, "rb");
    if (!warmup_file) {
        fprintf(stderr, "Error: Cannot open warmup file %s\n", warmup_filename);
        free(lpc_coefficients);
        return 1;
    }
    
    /* Get warmup file size */
    fseek(warmup_file, 0, SEEK_END);
    long warmup_file_size = ftell(warmup_file);
    fseek(warmup_file, 0, SEEK_SET);
    
    if (warmup_file_size != order * sizeof(float)) {
        fprintf(stderr, "Error: Warmup file size %ld doesn't match expected size %zu\n", 
                warmup_file_size, order * sizeof(float));
        free(lpc_coefficients);
        fclose(warmup_file);
        return 1;
    }
    
    /* Read warmup samples */
    float *warmup_samples = malloc(order * sizeof(float));
    if (!warmup_samples) {
        fprintf(stderr, "Error: Memory allocation failed for warmup samples\n");
        free(lpc_coefficients);
        fclose(warmup_file);
        return 1;
    }
    
    if (fread(warmup_samples, sizeof(float), order, warmup_file) != (size_t)order) {
        fprintf(stderr, "Error: Failed to read warmup samples\n");
        free(lpc_coefficients);
        free(warmup_samples);
        fclose(warmup_file);
        return 1;
    }
    fclose(warmup_file);
    
    /* Allocate output array */
    float *predicted_samples = malloc(n_predicted_samples * sizeof(float));
    if (!predicted_samples) {
        fprintf(stderr, "Error: Memory allocation failed for predicted samples\n");
        free(lpc_coefficients);
        free(warmup_samples);
        return 1;
    }
    
    /* Perform LPC prediction */
    if (lpc_predict_samples(lpc_coefficients, order, warmup_samples, n_predicted_samples, predicted_samples) != 0) {
        fprintf(stderr, "Error: LPC prediction failed\n");
        free(lpc_coefficients);
        free(warmup_samples);
        free(predicted_samples);
        return 1;
    }
    
    /* Output predicted samples to stdout */
    fwrite(predicted_samples, sizeof(float), n_predicted_samples, stdout);
    
    /* Cleanup */
    free(lpc_coefficients);
    free(warmup_samples);
    free(predicted_samples);
    
    return 0;
} 