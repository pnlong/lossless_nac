# Zero-Shot Language Model Evaluation on Custom Audio Data

## Overview

This document outlines how to evaluate a pre-trained language model's compression performance on custom audio data without training, using the language_modeling_is_compression framework.

## Key Components for Zero-Shot Evaluation

### 1. Main Evaluation Script: `compress.py`

The primary evaluation script is located at `/language_modeling_is_compression/compress.py`. This script:

- **Evaluates compressors** on different datasets without training
- **Supports multiple compressor types**: classical (gzip, lzma, flac, png) and arithmetic coding (language_model)
- **Handles chunked vs unchunked evaluation** for different data types
- **Provides compression ratios and timing metrics**

### 2. Custom Audio Data Integration

The framework already includes your custom audio iterator in `data_loaders.py`:

```python
def get_custom_audio_iterator(
    audio_files: List[str],
    num_chunks: int = constants.NUM_CHUNKS,
    use_16bit: bool = False,
    blocking_size: int = 1024,
    chunk_size_bytes: int = constants.CHUNK_SIZE_BYTES,
) -> Iterator[bytes]:
```

**Key features:**
- Supports both 8-bit (vocab size 256) and 16-bit (vocab size 65536) audio
- Handles stereo blocking strategy for multi-channel audio
- Configurable chunk sizes and blocking parameters
- Integrated into the `GET_DATA_GENERATOR_FN_DICT` as `'custom_audio'`

### 3. Language Model Compression Interface

The language model compressor is implemented in `compressors/language_model.py`:

```python
def compress(
    data: bytes,
    return_num_padded_bits: bool = False,
    use_slow_lossless_compression: bool = False,
    use_16bit: bool = False,
) -> bytes | tuple[bytes, int]:
```

**Key capabilities:**
- Uses arithmetic coding with pre-trained model predictions
- Supports both 8-bit and 16-bit audio modes
- Handles model parameter loading from `params.npz`
- Provides both compression and decompression functions

## How to Run Zero-Shot Evaluation

### Step 1: Prepare Pre-trained Model

You need a pre-trained model saved as `params.npz` in the working directory. This can be obtained by:

1. **Training a model** using your `train_audio.py` script
2. **Using a pre-trained model** from the original paper (if available)
3. **Training on enwik8** using the original `train.py` script

### Step 2: Set Up Environment

```bash
# Activate conda environment
conda activate lmic

# Set PYTHONPATH
export PYTHONPATH=$(pwd)/..

# Navigate to the language_modeling_is_compression directory
cd language_modeling_is_compression
```

### Step 3: Run Evaluation

#### Basic Command Structure:
```bash
python compress.py --compressor language_model --dataset custom_audio --num_chunks 1000
```

#### For 16-bit Audio:
```bash
python compress.py --compressor language_model --dataset custom_audio --num_chunks 1000 --use_16bit
```

#### With Custom Audio Files:
You'll need to modify the `get_custom_audio_iterator` call in `compress.py` to pass your audio file paths, or create a wrapper script.

## Implementation Details

### Audio Processing Pipeline

1. **Data Loading**: WAV files are loaded using `scipy.io.wavfile`
2. **Format Conversion**: Audio is normalized to [-1, 1] range and converted to target bit depth
3. **Stereo Blocking**: For stereo audio, blocks are interleaved as [L_block1, R_block1, L_block2, R_block2, ...]
4. **Chunking**: Audio is split into fixed-size chunks (default 2048 bytes)
5. **Compression**: Each chunk is compressed using arithmetic coding with language model predictions

### Model Configuration

The transformer model automatically adjusts its vocabulary size based on the `use_16bit` parameter:

- **8-bit mode**: vocab_size = 256 (ALPHABET_SIZE)
- **16-bit mode**: vocab_size = 65536 (ALPHABET_SIZE_16BIT)

### Compression Evaluation

The evaluation process:

1. **Loads pre-trained model parameters** from `params.npz`
2. **Processes audio data** through the custom iterator
3. **Applies arithmetic coding** using model predictions
4. **Calculates compression ratio** as compressed_size / original_size
5. **Reports timing metrics** for performance analysis

## Expected Results

### Compression Performance

Based on the paper's findings:
- **Chinchilla 70B** compresses LibriSpeech to 16.4% of original size
- **Language models** generally outperform domain-specific compressors on audio data
- **16-bit audio** may show different compression characteristics than 8-bit

### Comparison Baselines

The framework provides several baseline compressors for comparison:
- **FLAC**: Domain-specific audio compressor (30.3% compression ratio on LibriSpeech)
- **GZIP**: General-purpose compressor
- **LZMA**: High-compression general-purpose compressor
- **PNG**: Image compressor (for reference)

## Customization Options

### Audio Processing Parameters

- `use_16bit`: Toggle between 8-bit and 16-bit audio processing
- `blocking_size`: Size of stereo blocks (default 1024 samples)
- `chunk_size_bytes`: Size of compression chunks (default 2048 bytes)
- `num_chunks`: Number of chunks to evaluate

### Model Architecture

The transformer configuration can be customized:
- `embedding_dim`: Embedding dimension (default 64)
- `num_layers`: Number of transformer layers (default 4)
- `num_heads`: Number of attention heads (default 8)
- `widening_factor`: Feedforward network scaling (default 4)

## Troubleshooting

### Common Issues

1. **Missing params.npz**: Train a model first or obtain pre-trained parameters
2. **Audio file format**: Ensure WAV files are properly formatted
3. **Memory issues**: Reduce `num_chunks` for large audio datasets
4. **Bit depth mismatch**: Ensure `use_16bit` parameter matches your audio data

### Performance Optimization

- Use GPU acceleration for faster model inference
- Adjust chunk sizes based on available memory
- Consider using `use_slow_lossless_compression=False` for faster evaluation

## Next Steps

1. **Implement audio file path integration** in the evaluation script
2. **Create wrapper scripts** for easy evaluation with custom audio directories
3. **Compare results** with baseline compressors (FLAC, GZIP, etc.)
4. **Analyze compression patterns** across different audio types and bit depths
5. **Extend evaluation** to include decompression accuracy testing
