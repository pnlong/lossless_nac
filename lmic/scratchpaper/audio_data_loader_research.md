# Audio Data Loader Research - LibriSpeech Analysis

## Overview

The `language_modeling_is_compression` codebase includes a working audio data loader for LibriSpeech that demonstrates how to convert raw audio waveforms into byte sequences suitable for transformer training. This analysis examines the existing implementation and identifies key modification points for adapting it to custom audio datasets (44.1kHz, mono/stereo).

## Current LibriSpeech Audio Processing Pipeline

### 1. Data Source and Format

**LibriSpeech Dataset Characteristics:**
- **Format**: TensorFlow Datasets (tfds) format
- **Sample Rate**: 16kHz (standard for speech)
- **Bit Depth**: 16-bit signed integers
- **Channels**: Mono (single channel)
- **Content**: Read English speech recordings
- **Split Used**: `train_clean100` (100 hours of clean speech)

**Data Access:**
```python
def _get_librispeech_dataset():
    return tfds.load('librispeech', split='train_clean100')
```

The dataset provides audio data in the `x['speech']` field as raw bytes representing 16-bit signed PCM audio.

### 2. Audio Preprocessing Pipeline

The preprocessing happens in `get_librispeech_iterator()` and involves several critical steps:

#### Step 1: Bit Depth Conversion (16-bit → 8-bit)
```python
audioop.lin2lin(x['speech'], 2, 1)
```
- **Input**: 16-bit signed samples (2 bytes per sample)
- **Output**: 8-bit samples (1 byte per sample)
- **Purpose**: Reduces vocabulary size from 65,536 to 256 possible values
- **Method**: Linear scaling from 16-bit range to 8-bit range

#### Step 2: Signed to Unsigned Conversion
```python
audioop.bias(..., 1, 128)
```
- **Input**: 8-bit signed samples (range: -128 to 127)
- **Output**: 8-bit unsigned samples (range: 0 to 255)
- **Purpose**: Maps to byte values (0-255) expected by the transformer
- **Method**: Adds 128 to shift the range from [-128, 127] to [0, 255]

#### Step 3: Chunking into Fixed-Size Sequences
```python
def _extract_audio_patches(sample: bytes) -> Iterator[bytes]:
    patches = np.array_split(
        np.frombuffer(sample, dtype=np.uint8),
        range(
            constants.CHUNK_SIZE_BYTES,  # 2048 bytes
            len(sample),
            constants.CHUNK_SIZE_BYTES,
        ),
    )
```

**Chunking Strategy:**
- **Chunk Size**: 2048 bytes (2048 samples at 8-bit)
- **Duration**: ~128ms at 16kHz (2048 samples / 16000 Hz)
- **Overlap**: No overlap between chunks
- **Padding**: Incomplete chunks at the end are discarded

### 3. Data Flow Summary

```
Raw LibriSpeech Audio (16-bit signed, 16kHz, mono)
    ↓
audioop.lin2lin(2, 1)  # 16-bit → 8-bit
    ↓
audioop.bias(1, 128)   # signed → unsigned (0-255)
    ↓
_extract_audio_patches()  # Split into 2048-byte chunks
    ↓
Byte sequences ready for transformer training
```

## Key Insights for Custom Audio Dataset Adaptation

### 1. Sample Rate Considerations

**Current Approach**: The existing pipeline processes audio at its native sample rate
**Your Requirement**: Support for various sample rates (44.1kHz, 48kHz, etc.)

**Solution**: No resampling required - process audio at native sample rates
- **Benefit**: No quality loss from resampling
- **Benefit**: Faster processing (no resampling overhead)
- **Benefit**: Simpler implementation

**Chunk Duration Analysis**:
- Fixed chunk size: 2048 bytes regardless of sample rate
- Duration varies by sample rate and bit depth:
  - 8-bit mono: 2048 samples = 46.4ms at 44.1kHz, 128ms at 16kHz
  - 16-bit mono: 1024 samples = 23.2ms at 44.1kHz, 64ms at 16kHz
  - 8-bit stereo: 1024 samples per channel = 23.2ms at 44.1kHz

### 2. Stereo Audio Handling

**Current Limitation**: Only handles mono audio
**Your Requirement**: Both mono and stereo support

**Stereo Processing Approach**: Interleaved Blocking Strategy

#### Blocking Pattern Implementation
```python
# Convert stereo to interleaved blocks
# [L_block1, R_block1, L_block2, R_block2, ...]
# Where each block contains 'blocking_size' samples
def process_stereo_blocking(audio_data, blocking_size):
    left_channel = audio_data[0]
    right_channel = audio_data[1]
    
    # Split into blocks and interleave
    blocks = []
    for i in range(0, len(left_channel), blocking_size):
        blocks.append(left_channel[i:i+blocking_size])
        blocks.append(right_channel[i:i+blocking_size])
    
    return np.concatenate(blocks)
```

**Benefits**:
- Maintains temporal relationship between L/R channels
- Configurable block size for different audio characteristics
- Handles partial blocks at the end of audio files

### 3. Bit Depth Considerations

**Current**: 16-bit → 8-bit conversion
**Alternatives**:
- **Keep 16-bit**: Requires vocabulary size 65536 (model architecture change)
- **Use 8-bit**: Maintains current 256 vocabulary
- **Use 12-bit**: Compromise with 4096 vocabulary

### 4. Chunking Strategy Modifications

**Current**: Fixed 2048-byte chunks
**For Stereo**: Need to consider channel alignment

**Stereo-Aware Chunking**:
```python
def _extract_stereo_audio_patches(sample: bytes, channels: int = 2) -> Iterator[bytes]:
    # Ensure chunks contain complete channel pairs
    samples_per_chunk = constants.CHUNK_SIZE_BYTES // channels
    # Align to channel boundaries
    aligned_chunk_size = (samples_per_chunk // channels) * channels
```

## Implementation Strategy for Custom Audio Loader

### 1. Required Modifications to `data_loaders.py`

```python
def get_custom_audio_iterator(
    audio_files: List[str],
    num_chunks: int = constants.NUM_CHUNKS,
    use_16bit: bool = False,
    blocking_size: int = 1024,
    chunk_size_bytes: int = constants.CHUNK_SIZE_BYTES,
) -> Iterator[bytes]:
    """Custom audio loader for WAV files with configurable bit depth and stereo blocking."""
    
    for audio_file in audio_files:
        # Load audio file (using scipy.io.wavfile)
        audio_data, sample_rate, num_channels = load_wav_file(audio_file)
        
        # Process stereo blocking
        processed_audio = process_stereo_blocking(audio_data, blocking_size)
        
        # Convert to target bit depth
        audio_bytes = convert_to_target_bit_depth(processed_audio, use_16bit)
        
        # Extract chunks
        for chunk in extract_audio_chunks(audio_bytes, chunk_size_bytes):
            yield chunk
```

### 2. Key Helper Functions Needed

```python
def load_wav_file(file_path: str) -> tuple[np.ndarray, int, int]:
    """Load WAV file and return (audio_data, sample_rate, num_channels)."""
    # Implementation using scipy.io.wavfile
    pass

def process_stereo_blocking(audio_data: np.ndarray, blocking_size: int) -> np.ndarray:
    """Process stereo audio using interleaved blocking strategy."""
    # Implementation for stereo blocking pattern
    pass

def convert_to_target_bit_depth(audio_data: np.ndarray, use_16bit: bool) -> bytes:
    """Convert audio data to target bit depth and return as bytes."""
    if use_16bit:
        # Convert to 16-bit signed integers
        audio_16bit = (audio_data * 32767).astype(np.int16)
        return audio_16bit.tobytes()
    else:
        # Convert to 8-bit unsigned integers
        audio_8bit = ((audio_data + 1.0) * 127.5).astype(np.uint8)
        return audio_8bit.tobytes()

def extract_audio_chunks(audio_bytes: bytes, chunk_size: int) -> Iterator[bytes]:
    """Extract fixed-size chunks from audio bytes."""
    # Implementation for chunking
    pass
```

### 3. Integration Points

**Update `GET_DATA_GENERATOR_FN_DICT`**:
```python
GET_DATA_GENERATOR_FN_DICT = {
    'enwik9': get_enwik9_iterator,
    'imagenet': get_imagenet_iterator,
    'librispeech': get_librispeech_iterator,
    'custom_audio': get_custom_audio_iterator,  # Add this
    'random': get_random_iterator,
}
```

**Modify `train.py`** to use custom audio loader:
```python
# In train_audio_transformer function
data_generator = data_loaders.get_custom_audio_iterator(
    audio_files=['path/to/audio1.wav', 'path/to/audio2.wav'],
    num_chunks=constants.NUM_CHUNKS,
    use_16bit=True,
    blocking_size=1024,
    chunk_size_bytes=constants.CHUNK_SIZE_BYTES,
)
```

## Considerations for Model Architecture

### 1. Vocabulary Size
- **Current**: 256 tokens (8-bit bytes)
- **For 16-bit audio**: 65536 tokens (requires model architecture change)
- **Recommendation**: Start with 8-bit to use existing model, then experiment with 16-bit

### 2. Sequence Length
- **Current**: 2048 bytes
- **For 44.1kHz stereo**: May need longer sequences for meaningful audio context
- **Consideration**: Balance between context length and computational efficiency

### 3. Positional Encoding
- **Current**: Sinusoidal encoding designed for text
- **For Audio**: May benefit from audio-specific positional encodings
- **Consideration**: Audio has different temporal characteristics than text

## Next Steps

1. **Implement custom audio loader** with support for 44.1kHz mono/stereo
2. **Test with small dataset** to validate the preprocessing pipeline
3. **Experiment with different chunk sizes** and bit depths
4. **Evaluate compression performance** compared to existing audio codecs
5. **Consider model architecture modifications** if needed for audio-specific patterns

The existing LibriSpeech implementation provides an excellent foundation for understanding how to convert raw audio into byte sequences suitable for transformer training. The key is adapting the preprocessing pipeline to handle your specific audio format requirements while maintaining compatibility with the existing model architecture.
