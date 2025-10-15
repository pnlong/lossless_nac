# Zero-Shot Evaluation Script Execution Flow

## **Step-by-Step Execution Flow**

### **Step 1: Entry Point**
```python
if __name__ == "__main__":
    main()
```
**What happens:**
- Script entry point calls `main()`.

### **Step 2: Main Function Setup**
```python
def main() -> None:
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(args.verbose)
        
        # Validate arguments
        validate_arguments(args)
        
        # Log configuration
        logging.info("Starting zero-shot language model evaluation")
        logging.info(f"Configuration: {vars(args)}")
        
        # Set random seed
        np.random.seed(args.seed)
```

**What happens:**
1. Parses CLI args.
2. Configures logging.
3. Validates args.
4. Logs config.
5. Seeds RNG.

### **Step 3: Argument Parsing**
```python
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(...)
    
    # Required arguments
    parser.add_argument("--audio_dir", required=True, ...)
    parser.add_argument("--model_path", required=True, ...)
    
    # Audio processing parameters
    parser.add_argument("--use_16bit", action="store_true", ...)
    parser.add_argument("--stereo_blocking_n", type=int, default=1024, ...)
    # ... more arguments
    
    return parser.parse_args()
```

**What happens:**
- Builds CLI interface.
- Parses args into `args`.
- Sets defaults.

### **Step 4: Logging Setup**
```python
def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Suppress absl logging if not verbose
    if not verbose:
        absl_logging.set_verbosity(absl_logging.ERROR)
```

**What happens:**
- Sets log level.
- Configures format/handler.
- Suppresses absl logs if not verbose.

### **Step 5: Argument Validation**
```python
def validate_arguments(args: argparse.Namespace) -> None:
    # Check if audio directory exists
    if not os.path.exists(args.audio_dir):
        raise ValueError(f"Audio directory does not exist: {args.audio_dir}")
    
    # Check if model file exists (only for framework models)
    model_type = detect_model_type(args.model_path)
    if model_type == "framework":
        if not os.path.exists(args.model_path):
            raise ValueError(f"Model file does not exist: {args.model_path}")
    
    # Validate positive integers
    positive_int_args = [...]
    for arg_name in positive_int_args:
        value = getattr(args, arg_name)
        if value <= 0:
            raise ValueError(f"{arg_name} must be positive, got {value}")
```

**What happens:**
- Verifies audio dir exists.
- Detects model type and validates accordingly.
- Ensures positive numeric args.

### **Step 6: Model Type Detection**
```python
def detect_model_type(model_path: str) -> str:
    if model_path.startswith("huggingface:"):
        return "huggingface"
    elif model_path.endswith(".npz"):
        return "framework"
    elif "/" in model_path and not os.path.exists(model_path):
        return "huggingface"
    else:
        raise ValueError(f"Unknown model format: {model_path}")
```

**What happens:**
- Returns "huggingface" for HF names or `huggingface:` prefix.
- Returns "framework" for `.npz` files.
- Otherwise errors.

### **Step 7: Comprehensive Evaluation**
```python
def run_comprehensive_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    results = {}
    
    # Set up audio data generator
    data_generator = setup_audio_data_generator(args)
    
    # Evaluate language model
    try:
        lm_results = evaluate_language_model(args.model_path, data_generator, args)
        results["language_model"] = lm_results
    except Exception as e:
        logging.error(f"Failed to evaluate language model: {str(e)}")
        results["language_model"] = {"error": str(e)}
    
    # Evaluate baseline compressors
    if args.baseline_compressors:
        baseline_results = evaluate_baseline_compressors(data_generator, args.baseline_compressors, args)
        results.update(baseline_results)
    
    return results
```

**What happens:**
- Builds audio data generator.
- Evaluates the language model.
- Evaluates baseline compressors if requested.
- Returns combined results.

### **Step 8: Audio Data Generator Setup**
```python
def setup_audio_data_generator(args: argparse.Namespace) -> Iterator[bytes]:
    # Get audio file paths
    try:
        audio_files = get_all_paths(args.audio_dir)
        logging.info(f"Found {len(audio_files)} WAV files in {args.audio_dir}")
    except Exception as e:
        raise ValueError(f"Error discovering audio files: {str(e)}")
    
    # Create data generator
    data_generator = data_loaders.get_custom_audio_iterator(
        audio_files=audio_files,
        num_chunks=args.num_chunks,
        use_16bit=args.use_16bit,
        blocking_size=args.stereo_blocking_n,
        chunk_size_bytes=args.chunk_size,
    )
    
    return data_generator
```

**What happens:**
- Discovers WAV files.
- Creates iterator with specified params.
- Returns generator yielding audio chunks.

### **Step 9: Language Model Evaluation**
```python
def evaluate_language_model(model_path: str, data_generator: Iterator[bytes], args: argparse.Namespace) -> Dict[str, Any]:
    logging.info("Evaluating language model...")
    
    # Load model parameters
    model_info = load_model_parameters(model_path, use_16bit=args.use_16bit, device=args.device)
    
    # Create prediction function based on model type
    if isinstance(model_info, dict) and model_info.get("model_type") == "huggingface":
        predict_fn = create_huggingface_predict_fn(model_info)
        model_type = "huggingface"
    else:
        predict_fn = create_model_predict_fn(model_info, use_16bit=args.use_16bit)
        model_type = "framework"
    
    # Create custom compression function
    def language_model_compress(data: bytes) -> bytes:
        # Convert data to array
        if args.use_16bit:
            sequence_array = np.frombuffer(data, dtype=np.int16)
        else:
            sequence_array = np.frombuffer(data, dtype=np.uint8)
        
        # Get predictions
        if args.slow_compression:
            log_probs = []
            for subsequence_length in range(len(sequence_array)):
                subsequence_probs = predict_fn(sequence_array[None, :subsequence_length + 1])
                log_probs.append(subsequence_probs[0, -1])
            log_probs = np.vstack(log_probs)
        else:
            log_probs = predict_fn(sequence_array[None])[0, ...]
        
        probs = np.exp(log_probs)
        
        # Use arithmetic coding
        from language_modeling_is_compression import arithmetic_coder
        
        output = []
        encoder = arithmetic_coder.Encoder(
            base=constants.ARITHMETIC_CODER_BASE,
            precision=constants.ARITHMETIC_CODER_PRECISION,
            output_fn=output.append,
        )
        
        for pdf, symbol in zip(probs, sequence_array):
            encoder.encode(utils.normalize_pdf_for_arithmetic_coding(pdf), symbol)
        encoder.terminate()
        
        compressed_bits = ''.join(map(str, output))
        compressed_bytes, _ = utils.bits_to_bytes(compressed_bits)
        
        return compressed_bytes
    
    # Evaluate compression
    start_time = time.perf_counter()
    compression_ratio, compression_time = evaluate_compressor_chunked(
        compress_fn=language_model_compress,
        get_data_generator_fn=lambda: data_generator,
        num_chunks=args.num_chunks,
        count_header_only_once=False,
        mask_fn=utils.right_shift_bytes_by_one,
        use_tqdm=args.use_tqdm,
    )
    total_time = time.perf_counter() - start_time
    
    return {
        "compression_ratio": compression_ratio,
        "compression_time": compression_time,
        "total_time": total_time,
        "compressor_type": "language_model",
        "model_type": model_type
    }
```

**What happens:**
- Loads model and builds prediction function.
- Defines `language_model_compress`:
  - Converts bytes to array.
  - Gets log-probs.
  - Applies arithmetic coding.
- Runs `evaluate_compressor_chunked` and returns metrics.

### **Step 10: Model Loading**
```python
def load_model_parameters(model_path: str, use_16bit: bool = False, device: str = "auto") -> Any:
    model_type = detect_model_type(model_path)
    
    if model_type == "huggingface":
        return load_huggingface_model(model_path, use_16bit, device)
    else:
        return load_framework_model(model_path)
```

**What happens:**
- Detects type and loads accordingly.
- Returns model info/params.

### **Step 11: Hugging Face Model Loading**
```python
def load_huggingface_model(model_name: str, use_16bit: bool = False, device: str = "auto") -> Dict[str, Any]:
    # Remove huggingface: prefix if present
    if model_name.startswith("huggingface:"):
        model_name = model_name.replace("huggingface:", "")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if use_16bit else torch.float32,
        device_map=device if device != "auto" else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set to evaluation mode
    model.eval()
    
    return {
        "model": model,
        "tokenizer": tokenizer,
        "model_type": "huggingface",
        "use_16bit": use_16bit
    }
```

**What happens:**
- Loads model and tokenizer.
- Sets pad token.
- Switches to eval mode.
- Returns model info dict.

### **Step 12: Baseline Compressor Evaluation**
```python
def evaluate_baseline_compressors(data_generator: Iterator[bytes], compressors: List[str], args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    results = {}
    
    for compressor_name in compressors:
        logging.info(f"Evaluating baseline compressor: {compressor_name}")
        
        try:
            compress_fn = compressor.COMPRESS_FN_DICT[compressor_name]
            
            # Create new data generator for each compressor
            audio_files = get_all_paths(args.audio_dir)
            new_data_generator = data_loaders.get_custom_audio_iterator(
                audio_files=audio_files,
                num_chunks=args.num_chunks,
                use_16bit=args.use_16bit,
                blocking_size=args.stereo_blocking_n,
                chunk_size_bytes=args.chunk_size,
            )
            
            start_time = time.perf_counter()
            
            if compressor_name in compressor.COMPRESSOR_TYPES['classical']:
                compression_ratio, compression_time = evaluate_compressor_chunked(
                    compress_fn=compress_fn,
                    get_data_generator_fn=lambda: new_data_generator,
                    num_chunks=args.num_chunks,
                    count_header_only_once=True,
                    mask_fn=None,
                    use_tqdm=args.use_tqdm,
                )
            else:
                # For arithmetic coding compressors
                compression_ratio, compression_time = evaluate_compressor_chunked(
                    compress_fn=compress_fn,
                    get_data_generator_fn=lambda: new_data_generator,
                    num_chunks=args.num_chunks,
                    count_header_only_once=False,
                    mask_fn=utils.right_shift_bytes_by_one,
                    use_tqdm=args.use_tqdm,
                )
            
            total_time = time.perf_counter() - start_time
            
            results[compressor_name] = {
                "compression_ratio": compression_ratio,
                "compression_time": compression_time,
                "total_time": total_time,
                "compressor_type": "baseline"
            }
            
        except Exception as e:
            logging.warning(f"Failed to evaluate {compressor_name}: {str(e)}")
            results[compressor_name] = {
                "error": str(e),
                "compressor_type": "baseline"
            }
    
    return results
```

**What happens:**
- Iterates baseline compressors.
- Builds fresh data generator per compressor.
- Runs evaluation and records metrics or errors.

### **Step 13: Result Formatting**
```python
def format_results(results: Dict[str, Any], args: argparse.Namespace) -> str:
    output = []
    output.append("Zero-Shot Language Model Evaluation Results")
    output.append("=" * 50)
    output.append("")
    
    # Configuration section
    output.append("Configuration:")
    output.append(f"  Audio Directory: {args.audio_dir}")
    output.append(f"  Model: {args.model_path}")
    output.append(f"  16-bit Audio: {args.use_16bit}")
    # ... more configuration details
    
    # Results section
    output.append("Results:")
    
    # Language model results
    if "language_model" in results:
        lm_result = results["language_model"]
        if "error" not in lm_result:
            ratio = lm_result["compression_ratio"] * 100
            time_taken = lm_result["compression_time"]
            model_type = lm_result.get("model_type", "framework")
            output.append(f"  Language Model ({model_type}): {ratio:.1f}% compression ratio ({time_taken:.1f}s)")
        else:
            output.append(f"  Language Model: ERROR - {lm_result['error']}")
    
    # Baseline results
    for compressor_name, result in results.items():
        if compressor_name != "language_model":
            if "error" not in result:
                ratio = result["compression_ratio"] * 100
                time_taken = result["compression_time"]
                output.append(f"  {compressor_name.upper()} Baseline: {ratio:.1f}% compression ratio ({time_taken:.1f}s)")
            else:
                output.append(f"  {compressor_name.upper()} Baseline: ERROR - {result['error']}")
    
    # Performance summary
    if "language_model" in results and "error" not in results["language_model"]:
        lm_ratio = results["language_model"]["compression_ratio"]
        best_compressor = "Language Model"
        best_ratio = lm_ratio
        
        # Find best baseline
        baseline_results = {k: v for k, v in results.items() if k != "language_model" and "error" not in v}
        if baseline_results:
            best_baseline = min(baseline_results.items(), key=lambda x: x[1]["compression_ratio"])
            if best_baseline[1]["compression_ratio"] < best_ratio:
                best_compressor = best_baseline[0].upper()
                best_ratio = best_baseline[1]["compression_ratio"]
        
        output.append("Performance Summary:")
        output.append(f"  Best Compressor: {best_compressor} ({best_ratio*100:.1f}%)")
        
        # Calculate improvements
        if baseline_results:
            for baseline_name, baseline_result in baseline_results.items():
                improvement = (baseline_result["compression_ratio"] - lm_ratio) / baseline_result["compression_ratio"]
                output.append(f"  Improvement over {baseline_name.upper()}: {improvement*100:.1f}% better")
        
        # Total time
        total_time = sum(r.get("total_time", 0) for r in results.values() if "error" not in r)
        output.append(f"  Total Evaluation Time: {total_time:.1f}s")
    
    return "\n".join(output)
```

**What happens:**
- Formats config, results, and summary.
- Computes best compressor and improvements.
- Returns formatted string.

### **Step 14: Result Output**
```python
# Format and display results
formatted_results = format_results(results, args)
print(formatted_results)

# Save results if requested
if args.output_file:
    save_results(results, args, args.output_file)

logging.info("Evaluation completed successfully!")
```

**What happens:**
- Prints formatted results.
- Optionally saves JSON.
- Logs completion.

## **Complete Execution Flow Summary**

1. **CLI parsing** → `parse_arguments()`
2. **Logging setup** → `setup_logging()`
3. **Validation** → `validate_arguments()`
4. **Model type detection** → `detect_model_type()`
5. **Audio generator** → `setup_audio_data_generator()`
6. **Model loading** → `load_model_parameters()`
7. **Language model evaluation** → `evaluate_language_model()`
8. **Baseline evaluation** → `evaluate_baseline_compressors()`
9. **Result formatting** → `format_results()`
10. **Output** → print and optional JSON save

The script evaluates language models on audio data and compares them to baseline compressors.

---

# How to Run Zero-Shot Evaluation with Different Llama Model Sizes

## Prerequisites

### 1. Install Dependencies
```bash
pip install transformers torch
```

### 2. Set Up Audio Data
Ensure you have a directory with WAV files:
```bash
# Example structure
/path/to/audio/
├── audio1.wav
├── audio2.wav
└── audio3.wav
```

### 3. Get Llama Model Access
Request access to Llama models from Meta AI:
- Visit: https://ai.facebook.com/blog/large-language-model-llama-meta-ai/
- Follow the instructions to request access
- Once approved, you'll receive download instructions

## Step-by-Step Evaluation Process

### Step 1: Quick Test with Framework Model
```bash
# Test with a framework model first (if you have one)
python zero_shot.py \
    --audio_dir /path/to/audio \
    --model_path "/path/to/your/model.npz" \
    --num_chunks 100 \
    --baseline_compressors gzip flac \
    --output_file framework_test.json \
    --verbose
```

### Step 2: Evaluate Different Llama Model Sizes
```bash
# Llama 2 7B (smallest)
python zero_shot.py \
    --audio_dir /path/to/audio \
    --model_path "./llama-2-7b-chat-hf" \
    --device "cuda:0" \
    --num_chunks 1000 \
    --baseline_compressors gzip flac lzma \
    --output_file llama2_7b_results.json \
    --verbose

# Llama 2 13B (medium)
python zero_shot.py \
    --audio_dir /path/to/audio \
    --model_path "./llama-2-13b-chat-hf" \
    --device "cuda:0" \
    --num_chunks 1000 \
    --baseline_compressors gzip flac lzma \
    --output_file llama2_13b_results.json \
    --verbose

# Llama 2 70B (largest)
python zero_shot.py \
    --audio_dir /path/to/audio \
    --model_path "./llama-2-70b-chat-hf" \
    --device "cuda:0" \
    --num_chunks 1000 \
    --baseline_compressors gzip flac lzma \
    --output_file llama2_70b_results.json \
    --verbose
```

### Step 3: Advanced Configuration for Large Models
```bash
# For very large models, use memory optimizations
python zero_shot.py \
    --audio_dir /path/to/audio \
    --model_path "./llama-2-70b-chat-hf" \
    --device "cuda:0" \
    --use_16bit \
    --max_length 2048 \
    --batch_size 8 \
    --num_chunks 500 \
    --baseline_compressors gzip flac lzma \
    --output_file llama2_70b_optimized.json \
    --verbose
```

## Comprehensive Comparison Script

### Use the Provided Evaluation Script
A ready-to-use evaluation script `evaluate_llama_models.sh` is available in the same directory as `zero_shot.py`. This script:

1. **Evaluates all three Llama model sizes** (7B, 13B, 70B)
2. **Compares against baseline compressors** (gzip, flac, lzma)
3. **Includes memory optimizations** for larger models
4. **Generates organized output files** for analysis

### Run the Evaluation Script
```bash
# Make the script executable (if not already done)
chmod +x evaluate_llama_models.sh

# Edit the script to set your audio directory path
# Then run it
./evaluate_llama_models.sh
```

### Manual Batch Evaluation (Alternative)
If you prefer to run evaluations manually or customize the process:

```bash
#!/bin/bash
# Manual evaluation script

AUDIO_DIR="/path/to/your/audio"
OUTPUT_DIR="./results"

# Create output directory
mkdir -p $OUTPUT_DIR

# Define Llama models to evaluate
MODELS=(
    "./llama-2-7b-chat-hf"
    "./llama-2-13b-chat-hf"
    "./llama-2-70b-chat-hf"
)

# Define baseline compressors
BASELINES="gzip flac lzma"

# Evaluate each model
for model_path in "${MODELS[@]}"; do
    echo "Evaluating model: $model_path"
    
    # Extract model name for filename
    model_name=$(basename "$model_path" | sed 's/-/_/g')
    
    python zero_shot.py \
        --audio_dir $AUDIO_DIR \
        --model_path "$model_path" \
        --device "cuda:0" \
        --num_chunks 1000 \
        --baseline_compressors $BASELINES \
        --output_file "$OUTPUT_DIR/${model_name}_results.json" \
        --verbose
done

echo "All evaluations completed!"
```

## Expected Output Format

### Console Output
```
Zero-Shot Language Model Evaluation Results
==========================================

Configuration:
  Audio Directory: /path/to/audio
  Model: ./llama-2-7b-chat-hf
  16-bit Audio: False
  Stereo Blocking: 1024 samples
  Chunk Size: 2048 bytes
  Number of Chunks: 1000

Results:
  Language Model (llama): 15.2% compression ratio (2.3s)
  FLAC Baseline: 28.7% compression ratio (0.8s)
  GZIP Baseline: 45.1% compression ratio (1.2s)
  LZMA Baseline: 38.9% compression ratio (2.1s)

Performance Summary:
  Best Compressor: Language Model (15.2%)
  Improvement over FLAC: 47.0% better
  Improvement over GZIP: 66.3% better
  Improvement over LZMA: 60.9% better
  Total Evaluation Time: 6.4s
```

### JSON Output
```json
{
  "configuration": {
    "audio_directory": "/path/to/audio",
    "model_path": "./llama-2-7b-chat-hf",
    "use_16bit": false,
    "stereo_blocking_n": 1024,
    "chunk_size": 2048,
    "num_chunks": 1000
  },
  "results": {
    "language_model": {
      "compression_ratio": 0.152,
      "compression_time": 2.3,
      "total_time": 2.3,
      "compressor_type": "language_model",
      "model_type": "llama"
    },
    "flac": {
      "compression_ratio": 0.287,
      "compression_time": 0.8,
      "total_time": 0.8,
      "compressor_type": "baseline"
    }
  },
  "summary": {
    "best_compressor": "language_model",
    "best_ratio": 0.152,
    "model_type": "llama",
    "improvements": {
      "vs_flac": 0.47,
      "vs_gzip": 0.663,
      "vs_lzma": 0.609
    },
    "total_evaluation_time": 6.4
  }
}
```

## Analysis and Comparison

### 1. Compare Results Across Model Sizes
```python
import json
import pandas as pd

# Load all results
results = {}
for model in ["llama_2_7b_chat_hf", "llama_2_13b_chat_hf", "llama_2_70b_chat_hf"]:
    with open(f"results/{model}_results.json", "r") as f:
        results[model] = json.load(f)

# Extract compression ratios
compression_ratios = {}
for model, data in results.items():
    if "error" not in data["results"]["language_model"]:
        compression_ratios[model] = data["results"]["language_model"]["compression_ratio"]

# Create comparison table
df = pd.DataFrame(list(compression_ratios.items()), columns=["Model", "Compression Ratio"])
df["Compression Ratio (%)"] = df["Compression Ratio"] * 100
df = df.sort_values("Compression Ratio")

print(df)
```

### 2. Plot Model Size vs Compression Performance
```python
import matplotlib.pyplot as plt

# Model sizes (approximate)
model_sizes = {
    "llama_2_7b_chat_hf": 7e9,
    "llama_2_13b_chat_hf": 13e9,
    "llama_2_70b_chat_hf": 70e9
}

# Plot
plt.figure(figsize=(10, 6))
for model, ratio in compression_ratios.items():
    plt.scatter(model_sizes[model], ratio * 100, label=model, s=100)

plt.xlabel("Model Size (Parameters)")
plt.ylabel("Compression Ratio (%)")
plt.title("Model Size vs Compression Performance")
plt.xscale("log")
plt.legend()
plt.grid(True)
plt.show()
```

## Tips for Successful Evaluation

### 1. Memory Management
- Use `--use_16bit` for large models to save memory
- Reduce `--batch_size` if you run out of GPU memory
- Use `--num_chunks` to control evaluation size

### 2. Performance Optimization
- Use `--device "cuda:0"` for GPU acceleration
- Adjust `--max_length` based on your audio chunk size
- Use `--verbose` for detailed progress tracking

### 3. Error Handling
- Start with small `--num_chunks` for testing
- Check GPU memory usage with `nvidia-smi`
- Monitor disk space for model downloads

### 4. Result Analysis
- Save results with `--output_file` for later analysis
- Compare against multiple baseline compressors
- Look for trends in compression performance vs model size

This approach will give you comprehensive insights into how different Llama model sizes affect zero-shot compression performance on your audio data.
