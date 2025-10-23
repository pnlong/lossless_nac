# Zero-Shot Language Model Evaluation Pipeline

This document provides a comprehensive explanation of the `zero_shot.py` script pipeline, detailing what happens from command-line argument parsing to the final "Evaluation completed successfully!" message.

## Overview

The `zero_shot.py` script evaluates pre-trained language models on custom audio data without training. It converts audio data to text sequences, uses language models to predict probabilities, and applies arithmetic coding for compression. The script supports both framework-based models (Haiku/JAX) and Llama models (PyTorch/Transformers).

## Pipeline Flow

### 1. Initialization and Setup

#### 1.1 Environment Configuration
```python
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # Line 16
```
- **Purpose**: Sets JAX to use CPU to avoid conflicts with PyTorch
- **Why**: Prevents GPU memory conflicts between JAX and PyTorch frameworks

#### 1.2 Module Imports
- **Core libraries**: `argparse`, `json`, `logging`, `numpy`, `haiku`
- **Framework components**: Imports from `language_modeling_is_compression` module
- **Lazy Llama imports**: PyTorch and Transformers are imported conditionally to avoid conflicts

### 2. Command-Line Argument Parsing (`parse_arguments()`)

#### 2.1 Required Arguments
- `--audio_dir`: Path to directory containing WAV files
- `--model_path`: Path to model file (.npz) or Llama model directory

#### 2.2 Audio Processing Parameters
- `--bit_depth`: Audio bit depth (8, 16, 24, or 32 bits) - **Default: 8**
- `--stereo_blocking_n`: Size of blocks for stereo processing - **Default: 1024**
- `--chunk_size`: Size of each compression chunk in bytes - **Default: 2048**
- `--num_chunks`: Maximum number of chunks to evaluate - **Default: 1000**

#### 2.3 Model Configuration
- `--embedding_dim`: Embedding dimension - **Default: 64**
- `--num_layers`: Number of transformer layers - **Default: 4**
- `--num_heads`: Number of attention heads - **Default: 8**
- `--widening_factor`: Feedforward network scaling factor - **Default: 4**

#### 2.4 Evaluation Parameters
- `--compressor`: Compressor to use - **Default: "language_model"**
- `--baseline_compressors`: List of baseline compressors - **Default: ['gzip', 'flac', 'lzma']**
- `--mixes_only`: Only include files where is_mix=true in CSV
- `--chunks_per_file`: Maximum chunks per file - **Default: 50**

### 3. Logging Setup (`setup_logging()`)

- **Verbose mode**: Sets logging level to DEBUG if `--verbose` flag is used
- **Format**: `%(asctime)s - %(levelname)s - %(message)s`
- **ABSL suppression**: Reduces absl logging verbosity unless verbose mode is enabled

### 4. Argument Validation (`validate_arguments()`)

#### 4.1 File System Checks
- Verifies `audio_dir` exists and is a directory
- Checks model file exists (for framework models)
- Validates CSV file exists at `dirname(audio_dir)/mixes.csv`

#### 4.2 Parameter Validation
- Ensures all integer parameters are positive
- Validates `chunk_size` is divisible by bytes per sample for given bit depth
- Warns if stereo blocking size is too small for chunk size
- Validates baseline compressor names

### 5. Model Type Detection (`detect_model_type()`)

The script automatically detects model type based on the path:

#### 5.1 Llama Models
- **Hugging Face format**: `config.json` + `pytorch_model.bin`
- **Original format**: `consolidated.00.pth` + `params.json`
- **Name-based**: Contains "llama" in path
- **Remote**: Contains "/" but doesn't exist locally

#### 5.2 Framework Models
- Files ending with `.npz` extension

### 6. Audio Data Setup (`setup_audio_data_generator()`)

#### 6.1 CSV-Based File Discovery
```python
csv_file = os.path.dirname(args.audio_dir) + "/mixes.csv"
audio_files = get_audio_files_from_csv(csv_file, args.audio_dir, args.mixes_only)
```
- Reads CSV file with columns: `path`, `is_mix`, `is_train`
- Filters by `is_mix=true` if `--mixes_only` flag is used
- Constructs full paths by joining `audio_dir` with basename of CSV paths

#### 6.2 Audio Pipeline Analysis
- Analyzes first audio file to detect potential issues:
  - **Silence detection**: Checks if audio amplitude is very low
  - **Repetitive data**: Warns if audio has few unique values
  - **Processing pipeline**: Tests the full audio processing chain

#### 6.3 Stereo Detection
```python
is_stereo = detect_stereo_audio(audio_files)
if is_stereo:
    effective_max_length = args.max_length * 2  # Double for stereo
```
- Checks first 5 audio files for stereo channels
- Adjusts `max_length` parameter for stereo audio (doubles sequence length)

#### 6.4 Data Generator Creation
```python
base_data_generator = audio_processing_extended.get_custom_audio_iterator_extended(
    audio_files=audio_files,
    num_chunks=args.num_chunks,
    bit_depth=args.bit_depth,
    blocking_size=args.stereo_blocking_n,
    chunk_size_bytes=args.chunk_size,
    chunks_per_file=args.chunks_per_file,
)
```
- Creates iterator that yields audio chunks as bytes
- Applies stereo blocking if needed
- Converts to target bit depth
- Samples chunks from each file

### 7. Model Loading (`load_model_parameters()`)

#### 7.1 Llama Model Loading (`load_llama_model()`)

The script supports **two different Llama formats**:

**Hugging Face Format** (if `config.json` + `pytorch_model.bin` exist):
```python
model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16)
tokenizer = LlamaTokenizer.from_pretrained(model_dir)
pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)
```

**Original Meta Format** (if `consolidated.00.pth` + `params.json` exist):
- Loads `params.json` for model configuration
- Loads `consolidated.00.pth` for model weights using `torch.load()`
- Uses `_run_original_llama_inference()` for inference (not Hugging Face)
- Creates wrapper for checkpoint data
- **This is what you're using** - your downloaded Llama models in original format

#### 7.2 Framework Model Loading (`load_framework_model()`)
```python
with np.load(model_path, allow_pickle=True) as data:
    params = {key: data[key].item() for key in data.files}
```
- Loads Haiku model parameters from `.npz` file

### 8. Language Model Evaluation (`evaluate_language_model()`)

#### 8.1 Prediction Function Creation
- **Llama models**: Uses `llama_integration.create_llama_predict_fn_extended()`
- **Framework models**: Uses `create_model_predict_fn()` with Haiku transformer

#### 8.2 Compression Function (`language_model_compress()`)

This is the core compression pipeline that processes each audio chunk:

**Step 1: Data Conversion**
```python
if args.bit_depth == 8:
    sequence_array = np.frombuffer(data, dtype=np.uint8)
elif args.bit_depth == 16:
    sequence_array = np.frombuffer(data, dtype=np.int16)
# ... similar for 24-bit and 32-bit
```

**Step 2: ASCII Mapping**
```python
ascii_mapping_fn = ascii_mapping.get_ascii_mapping_function_for_bit_depth(args.bit_depth)
ascii_data, dropped_lsb_bits = ascii_mapping_fn(data_bytes)
```
- Converts binary data to ASCII characters
- Preserves least significant bits for lossless reconstruction
- Mapping ratios: 8-bit→1 char, 16-bit→2 chars, 24-bit→3 chars, 32-bit→4 chars

**Step 3: Text Generation**
```python
ascii_text = ascii_data.decode('ascii', errors='ignore')
log_probs = predict_fn(ascii_text)  # Get language model predictions
```

**Step 4: Tokenization**
```python
tokens = model_info["tokenizer"].encode(ascii_text, add_special_tokens=False)
```

**Step 5: Top-K Filtering**
```python
log_probs_topk = llama_integration.apply_top_k_filtering(log_probs, k=100)
```
- Applies top-k filtering with k=100 as described in the paper
- Handles tokens not in top-k with uniform distribution over remaining vocabulary

**Step 6: Arithmetic Coding**
```python
encoder = arithmetic_coder.Encoder(base=constants.ARITHMETIC_CODER_BASE, precision=constants.ARITHMETIC_CODER_PRECISION)
for i, token_id in enumerate(tokens):
    pdf = np.exp(log_probs_topk[i])  # Convert to probabilities
    encoder.encode(utils.normalize_pdf_for_arithmetic_coding(pdf), token_id)
```

**Step 7: Final Compression**
```python
compressed_bits = ''.join(map(str, output))
compressed_bytes, _ = utils.bits_to_bytes(compressed_bits)
final_compressed = compressed_bytes + dropped_lsb_bits
```

#### 8.3 Compression Evaluation
```python
compression_ratio, compression_time = evaluate_compressor_chunked(
    compress_fn=language_model_compress,
    get_data_generator_fn=lambda: data_generator,
    num_chunks=args.num_chunks,
    count_header_only_once=False,
    mask_fn=mask_fn,
    use_tqdm=True,
)
```

### 9. Baseline Compressor Evaluation (`evaluate_baseline_compressors()`)

#### 9.1 Compressor Types
- **Classical**: gzip, flac, lzma
- **Arithmetic coding**: Uses different mask functions

#### 9.2 Evaluation Process
- Creates new data generator for each compressor
- Uses same audio processing pipeline
- Applies appropriate mask functions
- Measures compression ratio and time

### 10. Results Processing

#### 10.1 Result Formatting (`format_results()`)
- **Configuration section**: Shows all parameters used
- **Results section**: Displays compression ratios and times
- **Performance summary**: Identifies best compressor and improvements

#### 10.2 Result Saving (`save_results()`)
- Saves detailed JSON results if `--output_file` specified
- Includes configuration, results, and summary statistics
- Calculates improvements over baseline compressors

### 11. Main Execution Flow (`main()`)

#### 11.1 Initialization Sequence
1. **Parse arguments** → `parse_arguments()`
2. **Setup logging** → `setup_logging()`
3. **Validate arguments** → `validate_arguments()`
4. **Set random seeds** → Ensures reproducibility

#### 11.2 Evaluation Sequence
1. **Run comprehensive evaluation** → `run_comprehensive_evaluation()`
   - Setup audio data generator
   - Evaluate language model
   - Evaluate baseline compressors
2. **Format and display results** → `format_results()`
3. **Save results** → `save_results()` (if requested)

#### 11.3 Completion
- Logs "Evaluation completed successfully!"
- Exits with error code if any step fails

## Key Technical Details

### Audio Processing Pipeline
1. **Load WAV files** using scipy.io.wavfile
2. **Stereo blocking** (if stereo): Interleaves left/right channels
3. **Normalization**: Converts to [-1, 1] range
4. **Bit depth conversion**: Converts to target bit depth
5. **Chunking**: Splits into fixed-size chunks

### Language Model Integration
- **Framework models**: Uses Haiku/JAX transformer architecture
- **Llama models**: Supports both Hugging Face format AND original Meta format
  - **Hugging Face**: Uses `LlamaForCausalLM.from_pretrained()` and Transformers pipeline
  - **Original Meta**: Uses `torch.load()` for `consolidated.00.pth` and custom inference
- **Prediction**: Generates log probabilities for next token prediction
- **Top-K filtering**: Limits vocabulary to top 100 tokens per position

### Compression Algorithm
- **Arithmetic coding**: Uses language model probabilities for encoding
- **Lossless reconstruction**: Preserves LSB bits for exact reconstruction
- **Variable length**: ASCII text length depends on bit depth

### Error Handling
- **Graceful degradation**: Continues evaluation even if some compressors fail
- **Comprehensive logging**: Detailed error messages and warnings
- **Validation**: Extensive input validation before processing

## Performance Considerations

### Memory Management
- **Lazy loading**: Llama modules loaded only when needed
- **Generator pattern**: Audio data processed in chunks
- **GPU optimization**: Uses float16 for Llama models

### Reproducibility
- **Random seeds**: Fixed seeds for numpy, random, and torch
- **Deterministic processing**: Consistent results across runs

### Scalability
- **Chunked processing**: Handles large audio datasets
- **Configurable parameters**: Adjustable chunk sizes and counts
- **Progress tracking**: Uses tqdm for progress bars

This pipeline provides a comprehensive framework for evaluating language models on audio compression tasks, with support for multiple model types, extensive configurability, and robust error handling.
