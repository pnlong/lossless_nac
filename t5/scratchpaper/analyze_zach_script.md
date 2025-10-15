# Analysis of T5 Audio Training Script (`train.py`)

## Overview
This script implements a T5-based sequence-to-sequence model for stereo audio channel prediction, where the model learns to predict one audio channel from another (L→R or R→L). The implementation uses PyTorch Lightning for training infrastructure and includes comprehensive logging capabilities.

## Data Loading Architecture

### 1. **StereoWavChunkDataset Class** (Lines 179-251)
The core dataset class that handles audio file loading and preprocessing:

**Key Components:**
- **File Discovery**: Uses `glob.glob()` to recursively find audio files with extensions: `['wav', 'flac', 'aiff', 'aif']`
- **Length Precomputation**: Loads precomputed file lengths from `musdbstereo_lengths.json` for efficient random sampling
- **Random Chunk Sampling**: Samples random fixed-length chunks from audio files using `torchaudio.load()` with `frame_offset` and `num_frames`

**Audio Processing Pipeline:**
```python
# 1. Load audio chunk
aud, sr = torchaudio.load(path, normalize=False, frame_offset=offset, num_frames=chunk_size, backend="soundfile")

# 2. Convert to 16-bit PCM if needed
if aud.dtype != torch.int16:
    aud = linear_encode(aud, bits=16)
else:
    aud = aud.long() + 32768  # map int16 -> uint16

# 3. Handle channel count
if aud.shape[0] < 2:
    aud = aud.repeat(2, 1)  # duplicate if mono
elif aud.shape[0] > 2:
    aud = aud[:2, :]  # take first two channels if more than 2

# 4. Randomly assign input/target channels
if torch.rand(1).item() < 0.5:
    input_ids = right; labels = left; order = 'R->L'
else:
    input_ids = left; labels = right; order = 'L->R'
```

### 2. **StereoDataModule Class** (Lines 255-320)
PyTorch Lightning DataModule that manages dataset splits and data loaders:

**Key Features:**
- **Deterministic Splits**: Uses seeded random shuffling for train/val/test splits
- **File Extension Support**: Supports multiple audio formats
- **Configurable Parameters**: Batch size, number of workers, chunk size, etc.

## Audio Format Handling

### Current Implementation Strengths:
1. **Multi-format Support**: Handles WAV, FLAC, AIFF, AIF files
2. **Automatic Channel Handling**: 
   - Mono → Duplicates to stereo
   - Multi-channel → Takes first two channels
   - Stereo → Uses as-is
3. **Flexible Bit Depth**: Converts non-16-bit audio to 16-bit PCM
4. **Random Channel Assignment**: Alternates between L→R and R→L prediction

### Quantization and Tokenization:
- **16-bit PCM Mapping**: Maps int16 samples to uint16 tokens by adding 32768
- **Vocabulary Size**: Default 65536 (2^16) for 16-bit samples
- **Alternative Quantization**: Includes mu-law and linear quantization functions (unused in current pipeline)

## Retrofitting for Different Audio Types

### 1. **Mono Audio Support**
The current implementation already handles mono audio by duplicating the channel:
```python
if aud.shape[0] < 2:
    aud = aud.repeat(2, 1)  # duplicate if mono
```

**For pure mono-to-mono prediction**, you could modify the dataset to:
- Skip channel duplication
- Use a different prediction task (e.g., predict future samples from past samples)
- Implement autoregressive prediction within a single channel

### 2. **Multi-channel Audio (Beyond Stereo)**
For 5.1, 7.1, or other multi-channel formats:

**Option A: Channel Selection**
```python
elif aud.shape[0] > 2:
    # Instead of just taking first two channels
    aud = aud[:2, :]  # current
    # Could select specific channels based on configuration
    aud = aud[[0, 1], :]  # front left, front right
    # or aud = aud[[2, 3], :]  # rear left, rear right
```

**Option B: Channel Aggregation**
```python
elif aud.shape[0] > 2:
    # Aggregate multiple channels into stereo
    front_channels = aud[:2, :]  # front left/right
    rear_channels = aud[2:4, :]  # rear left/right
    aud = torch.stack([
        front_channels.mean(dim=0),  # aggregated left
        rear_channels.mean(dim=0)   # aggregated right
    ])
```

### 3. **Different Sample Rates**
The current implementation assumes 44.1kHz. To support different sample rates:

**Modify the dataset constructor:**
```python
def __init__(self, file_list: List[str], chunk_size: int, sample_rate: int = 44100, ...):
    # Add resampling if needed
    if sr != self.sample_rate:
        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
        aud = resampler(aud)
```

### 4. **Different Bit Depths**
The current implementation hardcodes 16-bit. To support 8-bit, 24-bit, or 32-bit:

**Modify the quantization logic:**
```python
def __init__(self, ..., bits: int = 16):
    self.bits = bits
    self.vocab_size = 1 << bits  # 2^bits

# In __getitem__:
if aud.dtype != torch.int16:
    aud = linear_encode(aud, bits=self.bits)
else:
    aud = aud.long() + (1 << (self.bits - 1))  # center around mid-point
```

### 5. **Different Audio Tasks**
Instead of channel prediction, you could implement:

**A. Temporal Prediction (Predict future from past):**
```python
# Split single channel into past/future
past_samples = aud[0, :chunk_size//2]
future_samples = aud[0, chunk_size//2:]
input_ids = past_samples
labels = future_samples
```

**B. Denoising (Predict clean from noisy):**
```python
# Add noise to clean audio
noisy_audio = clean_audio + noise
input_ids = noisy_audio
labels = clean_audio
```

**C. Compression/Decompression:**
```python
# Compress audio (e.g., lower bit rate)
compressed = compress_audio(original_audio)
input_ids = compressed
labels = original_audio
```

## Key Dependencies and Requirements

### External Dependencies:
- `torchaudio`: Audio loading and processing
- `soundfile`: Alternative audio backend
- `pytorch_lightning`: Training framework
- `transformers`: T5 model implementation
- `wandb`: Experiment logging (optional)

### File Requirements:
- `musdbstereo_lengths.json`: Precomputed file lengths for efficient sampling
- Audio files in supported formats (WAV, FLAC, AIFF, AIF)

## Performance Considerations

### Memory Usage:
- **Sequence Length**: O(L^2) memory complexity for transformer attention
- **Vocabulary Size**: 65536 tokens × hidden_dim for final linear layer
- **Batch Size**: Configurable but limited by GPU memory

### Optimization Opportunities:
1. **Reduce Vocabulary**: Use mu-law quantization (256 tokens) instead of 16-bit (65536 tokens)
2. **Hierarchical Processing**: Use convolutional front-end to downsample before transformer
3. **Chunk Size**: Balance between context length and memory usage
4. **LoRA**: Already implemented for parameter-efficient fine-tuning

## Recommendations for Extension

1. **Make Audio Processing Configurable**: Add parameters for bit depth, sample rate, and channel handling
2. **Support More Audio Formats**: Add support for MP3, M4A, OGG, etc.
3. **Implement Alternative Tasks**: Add support for different prediction tasks beyond channel prediction
4. **Add Data Augmentation**: Implement audio augmentation techniques (pitch shift, time stretch, noise addition)
5. **Improve Error Handling**: Add better error handling for corrupted or unsupported audio files
6. **Optimize Memory Usage**: Implement streaming for very large audio files

The current implementation provides a solid foundation for stereo audio channel prediction and can be easily extended to support different audio types and prediction tasks with minimal modifications to the core architecture.
