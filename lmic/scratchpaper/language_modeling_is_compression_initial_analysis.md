# Language Modeling is Compression - Codebase Analysis

## Overview

This codebase implements the "Language Modeling is Compression" approach from the ICLR 2024 paper. The core idea is that large language models can be used as powerful general-purpose compressors by combining them with arithmetic coding. The framework treats any data (text, images, audio) as sequences of bytes and uses a trained transformer to predict the next byte, then compresses using arithmetic coding.

## Key Components

### 1. Data Loading Framework (`data_loaders.py`)

The data loading system is designed to handle different data types by converting them to byte sequences:

**Current Data Types Supported:**
- **Enwik9**: Raw text data (Wikipedia dump)
- **LibriSpeech**: Audio data converted to 8-bit mono
- **ImageNet**: Images converted to grayscale patches
- **Random**: Random byte sequences for baseline

**Key Data Loading Patterns:**
- All data is chunked into fixed-size byte sequences (`CHUNK_SIZE_BYTES = 2048`)
- Audio data is preprocessed: 16-bit → 8-bit conversion, stereo → mono, signed → unsigned
- Images are converted to grayscale and split into 32x64 patches
- Data is returned as iterators of bytes objects

**Audio Processing (LibriSpeech):**
```python
# Convert 16-bit signed to 8-bit unsigned
audioop.bias(audioop.lin2lin(x['speech'], 2, 1), 1, 128)
```

### 2. Transformer Architecture (`transformer.py`)

**Model Configuration:**
- **Architecture**: Decoder-only transformer (GPT-style)
- **Vocabulary Size**: 256 (byte-level, one for each possible byte value)
- **Default Config**: 4 layers, 8 heads, 64 embedding dim, 4x widening factor
- **Position Encoding**: Sinusoidal (from original transformer paper)
- **Activation**: GELU
- **Normalization**: Layer norm after attention and FFN

**Key Features:**
- Right-shifts input sequences for causal prediction
- Uses causal masking for autoregressive generation
- Outputs log-softmax probabilities over 256 possible byte values
- Designed for byte-level prediction tasks

### 3. Compression Framework (`compressors/`)

**Compressor Interface:**
- All compressors implement a simple protocol: `(data: bytes) -> bytes`
- Two categories: classical (FLAC, PNG, gzip, lzma) and arithmetic coding (language models)

**Language Model Compressor (`language_model.py`):**
- Loads trained transformer parameters from `params.npz`
- Uses arithmetic coding with the model's probability predictions
- Supports both fast (O(n)) and slow (O(n²)) compression modes
- Includes decompression functionality

**Arithmetic Coding (`arithmetic_coder.py`):**
- Implements standard arithmetic coding with configurable base and precision
- Base 2 (binary), 32-bit precision by default
- Handles carry digits and normalization for numerical stability

### 4. Training Process (`train.py`)

**Training Setup:**
- Trains on Enwik8 data (subset of Enwik9)
- Uses Adam optimizer with 1e-4 learning rate
- Batch size 128, sequence length 2048 bytes
- Loss: negative log-likelihood of true bytes
- Gradient normalization by sequence length

**Key Training Details:**
- Converts byte sequences to uint8 arrays
- Uses JAX/Haiku for efficient training
- Saves parameters to `params.npz` for later use in compression

## Constants and Configuration (`constants.py`)

```python
NUM_CHUNKS = 488281          # Number of data chunks
CHUNK_SIZE_BYTES = 2048      # Fixed chunk size in bytes
CHUNK_SHAPE_2D = (32, 64)    # For 2D data (images)
ALPHABET_SIZE = 256          # Byte vocabulary size
ARITHMETIC_CODER_BASE = 2    # Binary arithmetic coding
ARITHMETIC_CODER_PRECISION = 32  # 32-bit precision
```

## Implications for Audio Compression

### Current Audio Handling
The codebase already includes LibriSpeech audio processing, but it's limited:
- **Mono only**: Converts stereo to mono
- **8-bit quantization**: Reduces from 16-bit to 8-bit
- **Fixed sample rate**: Assumes 16kHz
- **Simple preprocessing**: Basic audioop operations

### Requirements for Custom Audio Data Loader

To support your mono/stereo audio compression needs, you'll need to create a custom data loader that:

1. **Handles Multiple Audio Formats**:
   - Support both mono and stereo audio
   - Handle different sample rates (16kHz, 44.1kHz, 48kHz, etc.)
   - Support different bit depths (16-bit, 24-bit, 32-bit)

2. **Audio Preprocessing**:
   - Convert to appropriate bit depth for the model (8-bit or 16-bit)
   - Handle stereo by either:
     - Interleaving L/R channels: `[L0, R0, L1, R1, L2, R2, ...]`
     - Concatenating channels: `[L0, L1, L2, ..., R0, R1, R2, ...]`
     - Processing channels separately

3. **Chunking Strategy**:
   - Current 2048-byte chunks may not align well with audio frames
   - Consider chunk sizes that align with audio frame boundaries
   - For stereo: ensure chunks contain complete L/R pairs

4. **Data Format Considerations**:
   - The transformer expects byte sequences (0-255)
   - Need to map audio samples to this range appropriately
   - Consider whether to use raw samples or some encoding

### Model Architecture Considerations

The current transformer is designed for byte-level prediction:
- **Vocabulary**: 256 tokens (one per byte)
- **Sequence Length**: 2048 bytes
- **Architecture**: Standard decoder-only transformer

For audio, you might consider:
- **Larger vocabulary**: If using 16-bit samples (65536 possible values)
- **Different chunk sizes**: Audio-specific sequence lengths
- **Specialized architectures**: Audio-specific attention patterns or positional encodings

### Integration Points

To integrate your audio data loader:

1. **Add to `data_loaders.py`**:
   ```python
   def get_custom_audio_iterator(
       audio_files: List[str],
       num_chunks: int = constants.NUM_CHUNKS,
       stereo: bool = True,
       sample_rate: int = 44100,
   ) -> Iterator[bytes]:
   ```

2. **Update `GET_DATA_GENERATOR_FN_DICT`**:
   ```python
   GET_DATA_GENERATOR_FN_DICT = {
       'enwik9': get_enwik9_iterator,
       'imagenet': get_imagenet_iterator,
       'librispeech': get_librispeech_iterator,
       'custom_audio': get_custom_audio_iterator,  # Add this
       'random': get_random_iterator,
   }
   ```

3. **Modify training loop** in `train.py` to use your data loader

4. **Consider model modifications** if needed for audio-specific requirements

## Next Steps

1. **Implement custom audio data loader** with support for mono/stereo
2. **Test with small audio dataset** to validate the approach
3. **Experiment with different chunk sizes** and preprocessing strategies
4. **Evaluate compression performance** compared to FLAC and other audio codecs
5. **Consider model architecture modifications** if needed for audio-specific patterns

The framework is well-designed and should be adaptable to your audio compression needs with the right data preprocessing and loading pipeline.
