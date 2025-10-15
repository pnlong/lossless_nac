# Running Zero-Shot Evaluation with Llama Models

## Prerequisites ✅

All dependencies are now installed in the `lmic` mamba environment:
- PyTorch 2.8.0 (with CUDA support)
- Transformers 4.57.0
- NumPy 2.1.3
- SciPy 1.15.3

## Step 1: Download Llama Models

You have several options for downloading Llama models:

### Option A: Download from Hugging Face (Recommended)

```bash
# Activate the environment
mamba activate lmic

# Download Llama 2 7B Chat model (smallest, good for testing)
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir ./llama-2-7b-chat-hf

# Or download Llama 2 13B Chat model (larger, better performance)
huggingface-cli download meta-llama/Llama-2-13b-chat-hf --local-dir ./llama-2-13b-chat-hf

# Or download Llama 3 8B Instruct model (newer)
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir ./llama-3-8b-instruct
```

### Option B: Use Existing Model Directory

If you already have Llama models downloaded, you can use them directly by pointing to the directory path.

## Step 2: Prepare Audio Data

Create a directory with your audio files (WAV format recommended):

```bash
# Create audio directory
mkdir -p /path/to/your/audio/data

# Copy your WAV files to this directory
# The system supports various bit depths: 8-bit, 16-bit, 24-bit, 32-bit
```

## Step 3: Run Zero-Shot Evaluation

### Basic Usage

```bash
# Activate environment
mamba activate lmic

# Navigate to the project directory
cd /home/pnlong/lnac/lmic

# Run zero-shot evaluation with 8-bit audio
python zero_shot.py \
    --audio_dir /path/to/your/audio/data \
    --model_path ./llama-2-7b-chat-hf \
    --bit_depth 8 \
    --num_chunks 100 \
    --verbose
```

### Different Bit Depths

```bash
# 8-bit audio (default, fastest)
python zero_shot.py \
    --audio_dir /path/to/your/audio/data \
    --model_path ./llama-2-7b-chat-hf \
    --bit_depth 8 \
    --num_chunks 100

# 16-bit audio (higher quality)
python zero_shot.py \
    --audio_dir /path/to/your/audio/data \
    --model_path ./llama-2-7b-chat-hf \
    --bit_depth 16 \
    --num_chunks 100

# 24-bit audio (professional quality)
python zero_shot.py \
    --audio_dir /path/to/your/audio/data \
    --model_path ./llama-2-7b-chat-hf \
    --bit_depth 24 \
    --num_chunks 100

# 32-bit audio (maximum quality)
python zero_shot.py \
    --audio_dir /path/to/your/audio/data \
    --model_path ./llama-2-7b-chat-hf \
    --bit_depth 32 \
    --num_chunks 100
```

### Advanced Options

```bash
# Full evaluation with baseline comparisons
python zero_shot.py \
    --audio_dir /path/to/your/audio/data \
    --model_path ./llama-2-7b-chat-hf \
    --bit_depth 16 \
    --num_chunks 1000 \
    --baseline_compressors gzip flac lzma \
    --chunk_size 4096 \
    --output_file results.json \
    --verbose

# Memory-optimized evaluation (for larger models)
python zero_shot.py \
    --audio_dir /path/to/your/audio/data \
    --model_path ./llama-2-13b-chat-hf \
    --bit_depth 8 \
    --num_chunks 100 \
    --device cpu \
    --verbose

# GPU-accelerated evaluation (if you have CUDA)
python zero_shot.py \
    --audio_dir /path/to/your/audio/data \
    --model_path ./llama-2-7b-chat-hf \
    --bit_depth 16 \
    --num_chunks 100 \
    --device cuda \
    --verbose
```

## Step 4: Understanding the Output

The evaluation will provide:

1. **Compression Ratio**: How well the Llama model compresses compared to the original data
2. **Compression Time**: How long the compression took
3. **Baseline Comparisons**: Comparison with traditional compressors (gzip, flac, lzma)
4. **Model Information**: Which model and bit depth was used

### Example Output:
```
Evaluating language model...
Found 50 WAV files in /path/to/your/audio/data
Loading Llama model from: ./llama-2-7b-chat-hf
Successfully loaded Llama model from ./llama-2-7b-chat-hf
Evaluating language model...
Language Model Results:
  Compression Ratio: 0.45 (45% of original size)
  Compression Time: 12.3 seconds
  Model Type: llama
  Bit Depth: 16

Baseline Compressor Results:
  gzip: 0.62 (62% of original size)
  flac: 0.38 (38% of original size)
  lzma: 0.55 (55% of original size)
```

## Step 5: Troubleshooting

### Common Issues and Solutions

1. **Out of Memory Error**:
   ```bash
   # Use smaller model or CPU
   python zero_shot.py --model_path ./llama-2-7b-chat-hf --device cpu --bit_depth 8
   ```

2. **Model Not Found**:
   ```bash
   # Check if model directory exists and has required files
   ls -la ./llama-2-7b-chat-hf/
   # Should contain: config.json, pytorch_model.bin, tokenizer files
   ```

3. **Audio Files Not Found**:
   ```bash
   # Check audio directory
   ls -la /path/to/your/audio/data/
   # Should contain .wav files
   ```

4. **Chunk Size Issues**:
   ```bash
   # For 16-bit audio, chunk_size must be even
   python zero_shot.py --bit_depth 16 --chunk_size 2048
   
   # For 24-bit audio, chunk_size must be divisible by 3
   python zero_shot.py --bit_depth 24 --chunk_size 3072
   
   # For 32-bit audio, chunk_size must be divisible by 4
   python zero_shot.py --bit_depth 32 --chunk_size 4096
   ```

## Step 6: Performance Tips

### For Faster Evaluation:
- Use `--bit_depth 8` (fastest processing)
- Use smaller models (7B instead of 13B/70B)
- Reduce `--num_chunks` for quick testing
- Use `--device cuda` if you have GPU

### For Better Compression:
- Use `--bit_depth 16` or higher (better quality)
- Use larger models (13B or 70B)
- Increase `--num_chunks` for more data
- Use `--slow_compression` for better accuracy

### For Memory Efficiency:
- Use `--device cpu` for CPU-only processing
- Use `--bit_depth 8` to reduce memory usage
- Reduce `--chunk_size` for smaller memory footprint

## Step 7: Example Commands

Here are some ready-to-use example commands:

```bash
# Quick test with small dataset
python zero_shot.py \
    --audio_dir /path/to/your/audio/data \
    --model_path ./llama-2-7b-chat-hf \
    --bit_depth 8 \
    --num_chunks 10 \
    --verbose

# Full evaluation with all baselines
python zero_shot.py \
    --audio_dir /path/to/your/audio/data \
    --model_path ./llama-2-7b-chat-hf \
    --bit_depth 16 \
    --num_chunks 1000 \
    --baseline_compressors gzip flac lzma \
    --output_file llama_results.json \
    --verbose

# High-quality evaluation
python zero_shot.py \
    --audio_dir /path/to/your/audio/data \
    --model_path ./llama-2-13b-chat-hf \
    --bit_depth 24 \
    --num_chunks 500 \
    --chunk_size 6144 \
    --verbose
```

## Next Steps

Once you have the basic evaluation working, you can:

1. **Compare different models**: Test 7B vs 13B vs 70B models
2. **Compare different bit depths**: See how quality affects compression
3. **Analyze results**: Use the JSON output for detailed analysis
4. **Scale up**: Run on larger datasets for more comprehensive evaluation

The implementation is now ready for production use with full Llama model support across all bit depths!

