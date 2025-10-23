# Fixing FLAC and Baseline Compressor Evaluation

## Problem Analysis

The current implementation has a fundamental design flaw: **all compressors (including FLAC) receive processed audio data**, when they should receive **raw audio data**. Only the language model should receive processed ASCII data.

### Current Issues

1. **FLAC gets processed data**: Normalized, chunked, bit-depth converted audio
2. **GZIP/LZMA get processed data**: Same processed audio chunks
3. **Language model gets processed data**: Correct (ASCII mapped)
4. **Unfair comparison**: Each compressor should get data in its optimal format

### Current Data Flow (BROKEN)
```
Raw WAV → Processed Audio Chunks → ALL COMPRESSORS
                              ↓
                    Language Model (ASCII)
```

### Desired Data Flow (FIXED)
```
Raw WAV → Raw Audio Chunks → FLAC/GZIP/LZMA
     ↓
Processed Audio → ASCII Mapping → Language Model
```

## Implementation Plan

### Phase 1: Create Raw Audio Data Generator

#### 1.1 New Function: `create_raw_audio_generator()`
```python
def create_raw_audio_generator(args: argparse.Namespace) -> Iterator[bytes]:
    """Create generator that yields raw audio chunks without processing.
    
    This generator provides raw audio data suitable for classical compressors
    like FLAC, GZIP, and LZMA.
    
    Args:
        args: Command line arguments
        
    Yields:
        Raw audio chunks as bytes (preserving original format)
    """
```

**Key Features:**
- Loads WAV files using `scipy.io.wavfile`
- Preserves original bit depth and format
- No normalization or processing
- No ASCII mapping
- Simple chunking by byte count
- Handles both mono and stereo (interleaved)

#### 1.2 Raw Audio Processing Logic
```python
# Load raw audio
sr, audio_data = wavfile.read(audio_file)

# Calculate samples per chunk based on target bit depth
bytes_per_sample = calculate_bytes_per_sample(args.bit_depth)
samples_per_chunk = args.chunk_size // bytes_per_sample

# Ensure we have complete sample chunks
total_samples = audio_data.size
complete_chunks = (total_samples // samples_per_chunk) * samples_per_chunk

# Truncate to complete chunks only
audio_data = audio_data[:complete_chunks]

# Convert to bytes preserving original format
if audio_data.dtype == np.int16:
    raw_bytes = audio_data.tobytes()
elif audio_data.dtype == np.int32:
    raw_bytes = audio_data.tobytes()
elif audio_data.dtype == np.uint8:
    raw_bytes = audio_data.tobytes()
else:
    # Convert to int16 as fallback
    raw_bytes = audio_data.astype(np.int16).tobytes()

# Chunk by sample count (not byte count)
for i in range(0, len(audio_data), samples_per_chunk):
    sample_chunk = audio_data[i:i + samples_per_chunk]
    if len(sample_chunk) == samples_per_chunk:
        # Convert sample chunk to bytes
        if sample_chunk.dtype == np.int16:
            chunk_bytes = sample_chunk.tobytes()
        elif sample_chunk.dtype == np.int32:
            chunk_bytes = sample_chunk.tobytes()
        elif sample_chunk.dtype == np.uint8:
            chunk_bytes = sample_chunk.tobytes()
        else:
            chunk_bytes = sample_chunk.astype(np.int16).tobytes()
        yield chunk_bytes
```

#### 1.3 Sample Count Validation
```python
def validate_sample_count_equivalence(raw_generator, processed_generator, args):
    """Validate that raw and processed generators yield equivalent sample counts."""
    raw_samples = 0
    processed_samples = 0
    
    # Count samples from raw generator
    for chunk in raw_generator:
        bytes_per_sample = calculate_bytes_per_sample(args.bit_depth)
        samples_in_chunk = len(chunk) // bytes_per_sample
        raw_samples += samples_in_chunk
    
    # Count samples from processed generator  
    for chunk in processed_generator:
        bytes_per_sample = calculate_bytes_per_sample(args.bit_depth)
        samples_in_chunk = len(chunk) // bytes_per_sample
        processed_samples += samples_in_chunk
    
    assert raw_samples == processed_samples, f"Sample count mismatch: raw={raw_samples}, processed={processed_samples}"
    logging.info(f"Sample count validation passed: {raw_samples} samples in both generators")
```

### Phase 2: Update Baseline Compressor Evaluation

#### 2.1 Modify `evaluate_baseline_compressors()`
```python
def evaluate_baseline_compressors(
    compressors: List[str],
    args: argparse.Namespace
) -> Dict[str, Dict[str, Any]]:
    """Evaluate baseline compressors on RAW audio data."""
    results = {}
    
    for compressor_name in compressors:
        logging.info(f"Evaluating baseline compressor: {compressor_name}")
        
        try:
            # Create RAW audio generator (not processed)
            raw_data_generator = create_raw_audio_generator(args)
            
            # Choose appropriate compression function
            if compressor_name == 'flac':
                compress_fn = create_flac_compress_fn(args.bit_depth)
            else:
                compress_fn = compressor.COMPRESS_FN_DICT[compressor_name]
            
            # Choose evaluation method based on compressor type
            if compressor_name == 'flac':
                # FLAC works best on continuous audio streams
                compression_ratio, compression_time = evaluate_compressor_unchunked(
                    compress_fn=compress_fn,
                    get_data_generator_fn=lambda: raw_data_generator,
                    num_chunks=args.num_chunks,
                )
            else:
                # GZIP/LZMA work fine on chunks
                compression_ratio, compression_time = evaluate_compressor_chunked(
                    compress_fn=compress_fn,
                    get_data_generator_fn=lambda: raw_data_generator,
                    num_chunks=args.num_chunks,
                    count_header_only_once=True,
                    mask_fn=None,  # No masking for raw data
                    use_tqdm=True,
                )
            
            results[compressor_name] = {
                "compression_ratio": compression_ratio,
                "compression_time": compression_time,
                "compressor_type": "baseline",
                "data_type": "raw_audio"
            }
            
        except Exception as e:
            logging.warning(f"Failed to evaluate {compressor_name}: {str(e)}")
            results[compressor_name] = {"error": str(e)}
    
    return results
```

#### 2.2 Create FLAC Compression Function
```python
def create_flac_compress_fn(bit_depth: int) -> Callable[[bytes], bytes]:
    """Create FLAC compression function with proper bit depth handling."""
    def flac_compress(data: bytes) -> bytes:
        # Import FLAC compressor
        from language_modeling_is_compression.compressors import flac
        return flac.compress(data, bit_depth=bit_depth)
    
    return flac_compress
```

### Phase 3: Update Language Model Evaluation

#### 3.1 Modify `evaluate_language_model()`
```python
def evaluate_language_model(
    model_path: str,
    args: argparse.Namespace
) -> Dict[str, Any]:
    """Evaluate language model on PROCESSED ASCII data."""
    logging.info("Evaluating language model...")
    
    # Create PROCESSED ASCII generator (existing functionality)
    processed_data_generator = setup_audio_data_generator(args)
    
    # Load model parameters
    model_info = load_model_parameters(model_path, use_gpu=args.gpu)
    
    # Create prediction function based on model type
    if isinstance(model_info, dict) and model_info.get("model_type") == "llama":
        from language_modeling_is_compression import llama_integration
        effective_max_length = getattr(args, 'effective_max_length', args.max_length)
        predict_fn = llama_integration.create_llama_predict_fn_extended(
            model_info, 
            bit_depth=args.bit_depth,
            max_length=effective_max_length
        )
        model_type = "llama"
    else:
        predict_fn = create_model_predict_fn(model_info)
        model_type = "framework"
    
    # Rest of existing language model evaluation code...
    # (language_model_compress function, etc.)
    
    return {
        "compression_ratio": compression_ratio,
        "compression_time": compression_time,
        "total_time": total_time,
        "compressor_type": "language_model",
        "model_type": model_type,
        "bit_depth": args.bit_depth,
        "data_type": "processed_ascii"
    }
```

### Phase 4: Update Main Evaluation Flow

#### 4.1 Modify `run_comprehensive_evaluation()`
```python
def run_comprehensive_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Run complete evaluation with proper data separation."""
    results = {}
    
    # Validate sample count equivalence between generators
    logging.info("Validating sample count equivalence...")
    raw_generator = create_raw_audio_generator(args)
    processed_generator = setup_audio_data_generator(args)
    validate_sample_count_equivalence(raw_generator, processed_generator, args)
    
    # Evaluate language model (uses processed ASCII data)
    try:
        lm_results = evaluate_language_model(args.model_path, args)
        results["language_model"] = lm_results
    except Exception as e:
        logging.error(f"Failed to evaluate language model: {str(e)}")
        results["language_model"] = {"error": str(e)}
    
    # Evaluate baseline compressors (uses raw audio data)
    if args.baseline_compressors:
        baseline_results = evaluate_baseline_compressors(args.baseline_compressors, args)
        results.update(baseline_results)
    
    return results
```

### Phase 5: Update Results Formatting

#### 5.1 Modify `format_results()` to Show Data Types
```python
def format_results(results: Dict[str, Any], args: argparse.Namespace) -> str:
    """Format evaluation results with data type information."""
    output = []
    output.append("Zero-Shot Language Model Evaluation Results")
    output.append("=" * 50)
    output.append("")
    
    # Configuration section
    output.append("Configuration:")
    output.append(f"  Audio Directory: {args.audio_dir}")
    output.append(f"  Model: {args.model_path}")
    output.append(f"  Bit Depth: {args.bit_depth}")
    output.append(f"  Chunk Size: {args.chunk_size} bytes")
    output.append("")
    
    # Results section with data type information
    output.append("Results:")
    
    # Language model results
    if "language_model" in results:
        lm_result = results["language_model"]
        if "error" not in lm_result:
            ratio = lm_result["compression_ratio"] * 100
            time_taken = lm_result["compression_time"]
            model_type = lm_result.get("model_type", "framework")
            data_type = lm_result.get("data_type", "processed_ascii")
            output.append(f"  Language Model ({model_type}): {ratio:.1f}% compression ratio ({time_taken:.1f}s) [Data: {data_type}]")
        else:
            output.append(f"  Language Model: ERROR - {lm_result['error']}")
    
    # Baseline results
    for compressor_name, result in results.items():
        if compressor_name != "language_model":
            if "error" not in result:
                ratio = result["compression_ratio"] * 100
                time_taken = result["compression_time"]
                data_type = result.get("data_type", "raw_audio")
                output.append(f"  {compressor_name.upper()} Baseline: {ratio:.1f}% compression ratio ({time_taken:.1f}s) [Data: {data_type}]")
            else:
                output.append(f"  {compressor_name.upper()} Baseline: ERROR - {result['error']}")
    
    return "\n".join(output)
```

## Expected Results After Implementation

### Before Fix (Current)
```
Language Model (llama): 217.1% compression ratio (165.3s)
GZIP Baseline: 95.3% compression ratio (0.0s)
FLAC Baseline: 461.7% compression ratio (24.6s)  ← TERRIBLE
LZMA Baseline: 89.7% compression ratio (0.1s)
```

### After Fix (Expected)
```
Language Model (llama): ~15-30% compression ratio (165.3s) [Data: processed_ascii]
GZIP Baseline: ~20-40% compression ratio (0.0s) [Data: raw_audio]
FLAC Baseline: ~10-25% compression ratio (24.6s) [Data: raw_audio]  ← MUCH BETTER
LZMA Baseline: ~15-30% compression ratio (0.1s) [Data: raw_audio]
```

## Implementation Steps

1. **Create `create_raw_audio_generator()` function with sample-based chunking**
2. **Create `validate_sample_count_equivalence()` function**
3. **Modify `evaluate_baseline_compressors()` to use raw data**
4. **Update `evaluate_language_model()` to use processed data**
5. **Modify `run_comprehensive_evaluation()` to validate sample equivalence**
6. **Update `format_results()` to show data types**
7. **Test with sample audio files**
8. **Verify sample count equivalence**
9. **Verify FLAC compression ratios improve significantly**

## Testing Strategy

1. **Unit tests** for raw audio generator with sample-based chunking
2. **Sample count validation tests** to ensure equivalence
3. **Integration tests** for each compressor type
4. **Comparison tests** before/after fix
5. **Performance tests** to ensure no regression
6. **Cross-validation tests** between raw and processed generators

## Risk Assessment

- **Low Risk**: Changes are isolated and well-defined
- **Backward Compatibility**: Existing functionality preserved
- **Performance Impact**: Minimal (same data processing, just different routing)
- **Testing**: Can verify improvements with existing test data

## Success Criteria

1. **FLAC compression ratio** improves from 461.7% to <50%
2. **Language model** continues to work on processed ASCII data
3. **GZIP/LZMA** performance remains similar or improves
4. **Sample count equivalence** validated between raw and processed generators
5. **Results clearly show** which compressors use which data types
6. **No regressions** in existing functionality
7. **Fair comparison** achieved with equivalent sample counts
