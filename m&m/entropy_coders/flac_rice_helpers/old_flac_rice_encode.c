#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define FFMIN(a, b) ((a) < (b) ? (a) : (b))

// FLAC type definitions
typedef int32_t FLAC__int32;
typedef uint32_t FLAC__uint32;
typedef uint64_t FLAC__uint64;
typedef int FLAC__bool;
#define true 1
#define false 0

// FLAC encoder configuration constants
#define FLAC_DEFAULT_PREDICTOR_ORDER 0              // No prediction for simplicity
#define FLAC_MAX_PARTITION_ORDER 8                  // RFC 9639 subset limit: <= 8
#define FLAC_MIN_PARTITION_ORDER 0                  // Always start from 0
#define FLAC_RICE_PARAMETER_LIMIT 30                // RICE2 supports up to 30 (escape at 31)
#define FLAC_RICE_PARAMETER_SEARCH_DIST 2           // Search distance for optimal rice parameter
#define FLAC_DEFAULT_BITS_PER_SAMPLE 16             // CD-quality audio standard
#define FLAC_DO_ESCAPE_CODING true                  // Enable escape coding for better compression

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
#define ENABLE_RICE_PARAMETER_SEARCH

// FLAC utility macros
#define flac_min(a,b) ((a) < (b) ? (a) : (b))
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

// Simple BitWriter implementation
typedef struct {
    uint8_t *buffer;
    size_t capacity;
    size_t bytes;
    uint32_t bits;
    uint32_t total_bits;
} BitWriter;

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

static void bitwriter_delete(BitWriter *bw) {
    if (bw) {
        free(bw->buffer);
        free(bw);
    }
}

static FLAC__bool bitwriter_write_raw_uint32(BitWriter *bw, uint32_t val, uint32_t bits) {
    if (bits == 0) return true;
    if (bw->bytes >= bw->capacity - 4) return false;
    
    bw->bits |= (val & ((1u << bits) - 1)) << (32 - bw->total_bits % 32 - bits);
    bw->total_bits += bits;
    
    while (bw->total_bits >= 8) {
        bw->buffer[bw->bytes++] = (bw->bits >> 24) & 0xFF;
        bw->bits <<= 8;
        bw->total_bits -= 8;
    }
    
    return true;
}

static FLAC__bool bitwriter_write_unary_unsigned(BitWriter *bw, uint32_t val) {
    for (uint32_t i = 0; i < val; i++) {
        if (!bitwriter_write_raw_uint32(bw, 1, 1)) return false;
    }
    return bitwriter_write_raw_uint32(bw, 0, 1);
}

static FLAC__bool bitwriter_write_rice_signed(BitWriter *bw, int32_t val, uint32_t parameter) {
    uint32_t uval = (val >= 0) ? (val << 1) : ((-val - 1) << 1) | 1;
    uint32_t quotient = uval >> parameter;
    uint32_t remainder = uval & ((1u << parameter) - 1);
    
    if (!bitwriter_write_unary_unsigned(bw, quotient)) return false;
    if (parameter > 0) {
        return bitwriter_write_raw_uint32(bw, remainder, parameter);
    }
    return true;
}

static FLAC__bool bitwriter_write_raw_int32(BitWriter *bw, int32_t val, uint32_t bits) {
    if (bits == 0) return true;
    if (bits > 32) return false;
    
    // Convert signed to unsigned preserving the bit pattern
    // The decoder will sign-extend based on the MSB
    uint32_t uval = (uint32_t)val & ((1u << bits) - 1);
    
    return bitwriter_write_raw_uint32(bw, uval, bits);
}

static void bitwriter_flush(BitWriter *bw) {
    if (bw->total_bits % 8 != 0) {
        bw->buffer[bw->bytes++] = (bw->bits >> 24) & 0xFF;
        bw->bits = 0;
        bw->total_bits = (bw->total_bits + 7) & ~7;
    }
}

// Bitmath function needed by FLAC
static uint32_t FLAC__bitmath_ilog2_wide(FLAC__uint64 v) {
    if (v == 0) return 0;
    uint32_t l = 0;
    while (v >>= 1) l++;
    return l;
}

// Copied from FLAC's stream_encoder.c
static void precompute_partition_info_sums_(
    const FLAC__int32 residual[],
    FLAC__uint64 abs_residual_partition_sums[],
    uint32_t residual_samples,
    uint32_t predictor_order,
    uint32_t min_partition_order,
    uint32_t max_partition_order,
    uint32_t bps __attribute__((unused))
) {
    const uint32_t default_partition_samples = (residual_samples + predictor_order) >> max_partition_order;
    uint32_t partitions = 1u << max_partition_order;
    
    // First do max_partition_order
    for(uint32_t i = 0; i < partitions; i++) {
        FLAC__uint64 abs_sum = 0;
        const FLAC__int32 *residual_ptr = residual + (i * default_partition_samples);
        for(uint32_t j = 0; j < default_partition_samples; j++) {
            const FLAC__int32 r = residual_ptr[j];
            abs_sum += (r < 0) ? -r : r;
        }
        abs_residual_partition_sums[i] = abs_sum;
    }
    
    // Now merge partitions for lower orders
    FLAC__uint64 *abs_residual_partition_sums_ptr = abs_residual_partition_sums + partitions;
    for(int32_t partition_order = max_partition_order - 1; partition_order >= (int32_t)min_partition_order; partition_order--) {
        partitions >>= 1;
        for(uint32_t i = 0; i < partitions; i++) {
            abs_residual_partition_sums_ptr[i] = 
                abs_residual_partition_sums[2*i] + 
                abs_residual_partition_sums[2*i + 1];
        }
        abs_residual_partition_sums = abs_residual_partition_sums_ptr;
        abs_residual_partition_sums_ptr += partitions;
    }
}

static void precompute_partition_info_escapes_(
    const FLAC__int32 residual[],
    uint32_t raw_bits_per_partition[],
    uint32_t residual_samples,
    uint32_t predictor_order,
    uint32_t min_partition_order,
    uint32_t max_partition_order
) {
    const uint32_t default_partition_samples = (residual_samples + predictor_order) >> max_partition_order;
    uint32_t *raw_bits_per_partition_ptr = raw_bits_per_partition;
    
    for(int32_t partition_order = max_partition_order; partition_order >= (int32_t)min_partition_order; partition_order--) {
        const uint32_t partitions = 1u << partition_order;
        const uint32_t partition_samples = default_partition_samples << (max_partition_order - partition_order);
        
        for(uint32_t i = 0; i < partitions; i++) {
            const FLAC__int32 *residual_ptr = residual + (i * partition_samples);
            uint32_t max_abs = 0;
            
            for(uint32_t j = 0; j < partition_samples; j++) {
                const FLAC__int32 r = residual_ptr[j];
                const uint32_t abs_val = (r < 0) ? -r : r;
                if(abs_val > max_abs)
                    max_abs = abs_val;
            }
            
            // Calculate minimum bits needed for signed representation
            uint32_t bits = 0;
            if(max_abs > 0) {
                // For signed integers, we need one extra bit for the sign
                // Calculate bits needed for the magnitude
                uint32_t temp = max_abs;
                while(temp) {
                    temp >>= 1;
                    bits++;
                }
                // Add one bit for sign
                bits++;
            }
            raw_bits_per_partition_ptr[i] = bits;
        }
        raw_bits_per_partition_ptr += partitions;
    }
}

// FLAC's count_rice_bits_in_partition_ function (exact version)
static inline uint32_t count_rice_bits_in_partition_(
    const uint32_t rice_parameter,
    const uint32_t partition_samples,
    const FLAC__int32 *residual
)
{
    uint32_t i;
    uint64_t partition_bits =
        FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_PARAMETER_LEN + /* actually could end up being FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE2_PARAMETER_LEN but err on side of 16bps */
        (1+rice_parameter) * partition_samples /* 1 for unary stop bit + rice_parameter for the binary portion */
    ;
    for(i = 0; i < partition_samples; i++)
        partition_bits += ( (FLAC__uint32)((residual[i]<<1)^(residual[i]>>31)) >> rice_parameter );
    return (uint32_t)(flac_min(partition_bits,UINT32_MAX)); // To make sure the return value doesn't overflow
}

// FLAC's set_partitioned_rice_ function
FLAC__bool set_partitioned_rice_(
    const FLAC__int32 residual[],
    const FLAC__uint64 abs_residual_partition_sums[],
    const uint32_t raw_bits_per_partition[],
    const uint32_t residual_samples,
    const uint32_t predictor_order,
    const uint32_t rice_parameter_limit,
    const uint32_t rice_parameter_search_dist,
    const uint32_t partition_order,
    const FLAC__bool search_for_escapes,
    FLAC__EntropyCodingMethod_PartitionedRiceContents *partitioned_rice_contents,
    uint32_t *bits
)
{
    uint32_t rice_parameter, partition_bits;
    uint32_t best_partition_bits, best_rice_parameter = 0;
    uint32_t bits_ = FLAC__ENTROPY_CODING_METHOD_TYPE_LEN + FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_ORDER_LEN;
    uint32_t *parameters, *raw_bits;
    uint32_t partition, residual_sample;
    uint32_t partition_samples, partition_samples_base;
    uint32_t partition_samples_fixed_point_divisor, partition_samples_fixed_point_divisor_base;
    const uint32_t partitions = 1u << partition_order;
    FLAC__uint64 mean;
#ifdef ENABLE_RICE_PARAMETER_SEARCH
    uint32_t min_rice_parameter, max_rice_parameter;
#else
    (void)rice_parameter_search_dist;
#endif

    FLAC__ASSERT(rice_parameter_limit <= FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE2_ESCAPE_PARAMETER);

    parameters = partitioned_rice_contents->parameters;
    raw_bits = partitioned_rice_contents->raw_bits;

    partition_samples_base = (residual_samples+predictor_order) >> partition_order;

    /* Check for division by zero - partition order too high for sample count */
    if (partition_samples_base == 0) {
        return false;
    }

    /* Integer division is slow. To speed up things, precalculate a fixed point
     * divisor, as all partitions except the first are the same size. 18 bits
     * are taken because maximum block size is 65535, max partition size for
     * partitions other than 0 is 32767 (15 bit), max abs residual is 2^31,
     * which leaves 18 bit */
    partition_samples_fixed_point_divisor_base = 0x40000 / partition_samples_base;

    for(partition = residual_sample = 0; partition < partitions; partition++) {
        partition_samples = partition_samples_base;
        if(partition > 0) {
            partition_samples_fixed_point_divisor = partition_samples_fixed_point_divisor_base;
        }
        else {
            if(partition_samples <= predictor_order)
                return false;
            else
                partition_samples -= predictor_order;
            
            /* Check for division by zero after subtracting predictor order */
            if (partition_samples == 0) {
                return false;
            }
            
            partition_samples_fixed_point_divisor = 0x40000 / partition_samples;
        }
        mean = abs_residual_partition_sums[partition];
        /* 'mean' is not a good name for the variable, it is
         * actually the sum of magnitudes of all residual values
         * in the partition, so the actual mean is
         * mean/partition_samples
         */
        if(mean < 2 || (((mean - 1)*partition_samples_fixed_point_divisor)>>18) == 0)
            rice_parameter = 0;
        else
            rice_parameter = FLAC__bitmath_ilog2_wide(((mean - 1)*partition_samples_fixed_point_divisor)>>18) + 1;

        if(rice_parameter >= rice_parameter_limit) {
            rice_parameter = rice_parameter_limit - 1;
        }

        best_partition_bits = UINT32_MAX;
#ifdef ENABLE_RICE_PARAMETER_SEARCH
        if(rice_parameter_search_dist) {
            if(rice_parameter < rice_parameter_search_dist)
                min_rice_parameter = 0;
            else
                min_rice_parameter = rice_parameter - rice_parameter_search_dist;
            max_rice_parameter = rice_parameter + rice_parameter_search_dist;
            if(max_rice_parameter >= rice_parameter_limit) {
                max_rice_parameter = rice_parameter_limit - 1;
            }
        }
        else
            min_rice_parameter = max_rice_parameter = rice_parameter;

        for(rice_parameter = min_rice_parameter; rice_parameter <= max_rice_parameter; rice_parameter++) {
#endif
            partition_bits = count_rice_bits_in_partition_(rice_parameter, partition_samples, residual+residual_sample);
            if(partition_bits < best_partition_bits) {
                best_rice_parameter = rice_parameter;
                best_partition_bits = partition_bits;
            }
#ifdef ENABLE_RICE_PARAMETER_SEARCH
        }
#endif
        if(search_for_escapes) {
            partition_bits = FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE2_PARAMETER_LEN + FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_RAW_LEN + raw_bits_per_partition[partition] * partition_samples;
            if(partition_bits <= best_partition_bits && raw_bits_per_partition[partition] < 32) {
                raw_bits[partition] = raw_bits_per_partition[partition];
                best_rice_parameter = 0; /* will be converted to appropriate escape parameter later */
                best_partition_bits = partition_bits;
            }
            else
                raw_bits[partition] = 0;
        }
        parameters[partition] = best_rice_parameter;
        if(best_partition_bits < UINT32_MAX - bits_) // To make sure _bits doesn't overflow
            bits_ += best_partition_bits;
        else
            bits_ = UINT32_MAX;
        residual_sample += partition_samples;
    }

    *bits = bits_;
    return true;
}

// Enhanced find_best_partition_order_ using FLAC's set_partitioned_rice_
static uint32_t find_best_partition_order_(
    const FLAC__int32 residual[],
    FLAC__uint64 abs_residual_partition_sums[],
    uint32_t raw_bits_per_partition[],
    uint32_t residual_samples,
    uint32_t predictor_order,
    uint32_t rice_parameter_limit,
    uint32_t min_partition_order,
    uint32_t max_partition_order,
    uint32_t bps,
    FLAC__bool do_escape_coding,
    uint32_t rice_parameter_search_dist,
    FLAC__EntropyCodingMethod *best_ecm
) {
    uint32_t best_partition_order = min_partition_order;
    uint32_t best_bits = UINT32_MAX;
    
    // Precompute partition info
    precompute_partition_info_sums_(
        residual,
        abs_residual_partition_sums,
        residual_samples,
        predictor_order,
        min_partition_order,
        max_partition_order,
        bps
    );
    
    if(do_escape_coding) {
        precompute_partition_info_escapes_(
            residual,
            raw_bits_per_partition,
            residual_samples,
            predictor_order,
            min_partition_order,
            max_partition_order
        );
    }
    
    const FLAC__uint64 *abs_residual_partition_sums_ptr = abs_residual_partition_sums;
    const uint32_t *raw_bits_per_partition_ptr = raw_bits_per_partition;
    
    // Skip to start of partition sums for min_partition_order
    for(uint32_t i = max_partition_order; i > min_partition_order; i--) {
        abs_residual_partition_sums_ptr += (1u << i);
        if(do_escape_coding)
            raw_bits_per_partition_ptr += (1u << i);
    }
    
    for(uint32_t partition_order = min_partition_order; partition_order <= max_partition_order; partition_order++) {
        uint32_t bits;
        FLAC__EntropyCodingMethod_PartitionedRiceContents test_contents;
        
        // Allocate temporary arrays for this partition order
        const uint32_t partitions = 1u << partition_order;
        test_contents.parameters = malloc(sizeof(uint32_t) * partitions);
        test_contents.raw_bits = malloc(sizeof(uint32_t) * partitions);
        
        if (!test_contents.parameters || !test_contents.raw_bits) {
            free(test_contents.parameters);
            free(test_contents.raw_bits);
            break;
        }
        
        // Use FLAC's set_partitioned_rice_ function
        if (set_partitioned_rice_(
            residual,
            abs_residual_partition_sums_ptr,
            raw_bits_per_partition_ptr,
            residual_samples,
            predictor_order,
            rice_parameter_limit,
            rice_parameter_search_dist,
            partition_order,
            do_escape_coding,
            &test_contents,
            &bits
        )) {
            if (bits < best_bits) {
                best_bits = bits;
                best_partition_order = partition_order;
                
                // Copy the best parameters
                memcpy(best_ecm->partitioned_rice.contents->parameters, 
                       test_contents.parameters, 
                       sizeof(uint32_t) * partitions);
                memcpy(best_ecm->partitioned_rice.contents->raw_bits, 
                       test_contents.raw_bits, 
                       sizeof(uint32_t) * partitions);
                
                // Set entropy coding method type
                FLAC__bool needs_rice2 = false;
                for(uint32_t i = 0; i < partitions; i++) {
                    if(test_contents.parameters[i] >= FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_ESCAPE_PARAMETER && test_contents.raw_bits[i] == 0) {
                        needs_rice2 = true;
                        break;
                    }
                }
                best_ecm->type = needs_rice2 ? FLAC__ENTROPY_CODING_METHOD_TYPE_PARTITIONED_RICE2 : FLAC__ENTROPY_CODING_METHOD_TYPE_PARTITIONED_RICE;
            }
        }
        
        free(test_contents.parameters);
        free(test_contents.raw_bits);
        
        abs_residual_partition_sums_ptr += partitions;
        if(do_escape_coding)
            raw_bits_per_partition_ptr += partitions;
    }
    
    best_ecm->partitioned_rice.order = best_partition_order;
    return best_partition_order;
}

// Write entropy coded data using BitWriter
static FLAC__bool write_entropy_coded_data(
    BitWriter *bw,
    const FLAC__int32 *residual,
    uint32_t residual_samples,
    uint32_t predictor_order,
    const FLAC__EntropyCodingMethod *ecm
) {
    const uint32_t partition_order = ecm->partitioned_rice.order;
    const uint32_t partitions = 1u << partition_order;
    const uint32_t *parameters = ecm->partitioned_rice.contents->parameters;
    const uint32_t *raw_bits = ecm->partitioned_rice.contents->raw_bits;
    const FLAC__bool is_rice2 = (ecm->type == FLAC__ENTROPY_CODING_METHOD_TYPE_PARTITIONED_RICE2);
    
    // Write entropy coding method type (2 bits)
    if (!bitwriter_write_raw_uint32(bw, ecm->type, FLAC__ENTROPY_CODING_METHOD_TYPE_LEN))
        return false;
    
    // Write partition order (4 bits)
    if (!bitwriter_write_raw_uint32(bw, partition_order, FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_ORDER_LEN))
        return false;
    
    uint32_t residual_sample = 0;
    const uint32_t default_partition_samples = (residual_samples + predictor_order) >> partition_order;
    
    // Check if partition order is too high for the number of samples
    if (default_partition_samples == 0) {
        return false;
    }
    
    for (uint32_t i = 0; i < partitions; i++) {
        uint32_t partition_samples = default_partition_samples;
        if (i == 0) {
            partition_samples -= predictor_order;
            if (partition_samples == 0) {
                return false;
            }
        }
        
        const uint32_t parameter = parameters[i];
        const uint32_t parameter_len = is_rice2 ? 
            FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE2_PARAMETER_LEN : 
            FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_PARAMETER_LEN;
        
        if (raw_bits[i] > 0) {
            // Escape coding
            const uint32_t escape_parameter = is_rice2 ? 
                FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE2_ESCAPE_PARAMETER : 
                FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_ESCAPE_PARAMETER;
            
            // Write escape parameter
            if (!bitwriter_write_raw_uint32(bw, escape_parameter, parameter_len))
                return false;
            
            // Write raw bits length (5 bits)
            if (!bitwriter_write_raw_uint32(bw, raw_bits[i], FLAC__ENTROPY_CODING_METHOD_PARTITIONED_RICE_RAW_LEN))
                return false;
            
            // Write raw residual values
            for (uint32_t j = 0; j < partition_samples; j++) {
                if (!bitwriter_write_raw_int32(bw, residual[residual_sample + j], raw_bits[i]))
                    return false;
            }
        } else {
            // Rice coding
            // Write rice parameter
            if (!bitwriter_write_raw_uint32(bw, parameter, parameter_len))
                return false;
            
            // Write rice-coded residual values
            for (uint32_t j = 0; j < partition_samples; j++) {
                if (!bitwriter_write_rice_signed(bw, residual[residual_sample + j], parameter))
                    return false;
            }
        }
        
        residual_sample += partition_samples;
    }
    
    return true;
}

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s input.bin [predictor_order] > output.bin\n", argv[0]);
        fprintf(stderr, "  input.bin: bitstream input file\n");
        fprintf(stderr, "  predictor_order: predictor order (optional, defaults to %d)\n", FLAC_DEFAULT_PREDICTOR_ORDER);
        return EXIT_FAILURE;
    }

    // Parse predictor order from command line or use default
    uint32_t predictor_order = FLAC_DEFAULT_PREDICTOR_ORDER;
    if (argc == 3) {
        char *endptr;
        long parsed_order = strtol(argv[2], &endptr, 10);
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

    // Get file size and calculate number of samples
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint32_t num_samples = file_size / sizeof(FLAC__int32);

    // Allocate memory for residuals
    FLAC__int32 *residuals = malloc(file_size);
    if (!residuals) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(fp);
        return EXIT_FAILURE;
    }

    // Read residuals
    if (fread(residuals, sizeof(FLAC__int32), num_samples, fp) != num_samples) {
        fprintf(stderr, "Error reading input file\n");
        free(residuals);
        fclose(fp);
        return EXIT_FAILURE;
    }
    fclose(fp);

    // Calculate appropriate partition order range like real FLAC encoders
    uint32_t blocksize = num_samples + predictor_order;
    
    // Calculate max partition order based on blocksize (ensure at least 16 samples per partition)
    uint32_t max_partition_order = 0;
    uint32_t temp_blocksize = blocksize;
    while (temp_blocksize > 16 && max_partition_order < FLAC_MAX_PARTITION_ORDER) {
        temp_blocksize >>= 1;
        max_partition_order++;
    }
    
    // For small blocksizes, use smaller partition orders  
    if (blocksize <= 192) {
        max_partition_order = FFMIN(max_partition_order, 2);
    } else if (blocksize <= 1152) {
        max_partition_order = FFMIN(max_partition_order, 3); 
    } else {
        max_partition_order = FFMIN(max_partition_order, 8);
    }
    
    // Ensure first partition has enough samples for the predictor
    if (predictor_order > 0) {
        uint32_t samples_in_first_partition = blocksize >> max_partition_order;
        while (samples_in_first_partition <= predictor_order && max_partition_order > 0) {
            max_partition_order--;
            samples_in_first_partition = blocksize >> max_partition_order;
        }
    }
    const uint32_t min_partition_order = FLAC_MIN_PARTITION_ORDER;
    const uint32_t rice_parameter_limit = FLAC_RICE_PARAMETER_LIMIT;
    const uint32_t rice_parameter_search_dist = FLAC_RICE_PARAMETER_SEARCH_DIST;
    const uint32_t bps = FLAC_DEFAULT_BITS_PER_SAMPLE;
    const FLAC__bool do_escape_coding = FLAC_DO_ESCAPE_CODING;

    // Calculate workspace sizes
    uint32_t sum_partition_order = 0;
    for(uint32_t i = min_partition_order; i <= max_partition_order; i++) {
        sum_partition_order += (1u << i);
    }

    // Allocate workspace arrays
    FLAC__uint64 *abs_residual_partition_sums = malloc(sizeof(FLAC__uint64) * sum_partition_order);
    uint32_t *raw_bits_per_partition = malloc(sizeof(uint32_t) * sum_partition_order);

    // Allocate FLAC entropy coding method structure
    FLAC__EntropyCodingMethod best_ecm;
    FLAC__EntropyCodingMethod_PartitionedRiceContents contents;
    contents.parameters = malloc(sizeof(FLAC__uint32) * (1u << max_partition_order));
    contents.raw_bits = malloc(sizeof(FLAC__uint32) * (1u << max_partition_order));
    best_ecm.partitioned_rice.contents = &contents;

    if (!abs_residual_partition_sums || !raw_bits_per_partition || 
        !contents.parameters || !contents.raw_bits) {
        fprintf(stderr, "Memory allocation failed\n");
        free(residuals);
        free(abs_residual_partition_sums);
        free(raw_bits_per_partition);
        free(contents.parameters);
        free(contents.raw_bits);
        return EXIT_FAILURE;
    }

    // Initialize best_ecm structure
    best_ecm.type = FLAC__ENTROPY_CODING_METHOD_TYPE_PARTITIONED_RICE;
    best_ecm.partitioned_rice.order = min_partition_order;
    
    // Use FLAC's enhanced rice coding algorithm
    find_best_partition_order_(
        residuals,
        abs_residual_partition_sums,
        raw_bits_per_partition,
        num_samples,
        predictor_order,
        rice_parameter_limit,
        min_partition_order,
        max_partition_order,
        bps,
        do_escape_coding,
        rice_parameter_search_dist,
        &best_ecm
    );

    // Create BitWriter for output
    BitWriter *bw = bitwriter_new(file_size * 20); // Allocate much more space for Rice coding expansion
    if (!bw) {
        fprintf(stderr, "Failed to create BitWriter\n");
        free(residuals);
        free(abs_residual_partition_sums);
        free(raw_bits_per_partition);
        free(contents.parameters);
        free(contents.raw_bits);
        return EXIT_FAILURE;
    }

    // Write entropy coded data using bit-oriented format
    if (!write_entropy_coded_data(bw, residuals, num_samples, predictor_order, &best_ecm)) {
        fprintf(stderr, "Failed to write entropy coded data\n");
        bitwriter_delete(bw);
        free(residuals);
        free(abs_residual_partition_sums);
        free(raw_bits_per_partition);
        free(contents.parameters);
        free(contents.raw_bits);
        return EXIT_FAILURE;
    }

    // Flush and write to stdout
    bitwriter_flush(bw);
    fwrite(bw->buffer, 1, bw->bytes, stdout);

    // Cleanup
    bitwriter_delete(bw);
    free(residuals);
    free(abs_residual_partition_sums);
    free(raw_bits_per_partition);
    free(contents.parameters);
    free(contents.raw_bits);

    return EXIT_SUCCESS;
}