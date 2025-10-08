# Audio Data Loader Implementation Plan

## Overview

This document outlines the implementation plan for a custom audio data loader that extends the existing LibriSpeech audio processing pipeline to support:
1. **16-bit audio support** with configurable bit depth (8-bit vs 16-bit)
2. **Stereo audio processing** with configurable blocking strategy
3. **WAV file input** from local file system with native sample rates (no resampling)
4. **Flexible data partitioning** for training

## Requirements Analysis

### 1. 16-bit Audio Support
- **Hyperparameter**: `use_16bit: bool`
- **When True**: Keep data in 16-bit space (65536 possible values)
- **When False**: Convert to 8-bit space (256 possible values, current behavior)
- **Model Impact**: 16-bit requires vocabulary size change from 256 to 65536

### 2. Stereo Audio Processing
- **Hyperparameter**: `blocking_size: int`
- **Strategy**: Interleaved blocking pattern
- **Pattern**: `[L_block1, R_block1, L_block2, R_block2, ...]`
- **Validation**: Warn if `blocking_size * 2 < chunk_size`

### 3. WAV File Input
- **Input**: List of WAV file paths
- **Support**: Both mono and stereo WAV files
- **Flexibility**: Handle different sample rates and bit depths (no resampling required)
- **Library**: Uses `scipy.io.wavfile` for efficient WAV file loading

## File Modification Plan

### 1. `data_loaders.py` - Primary Modifications

#### New Function: `get_custom_audio_iterator()`
```python
def get_custom_audio_iterator(
    audio_files: List[str],
    num_chunks: int = constants.NUM_CHUNKS,
    use_16bit: bool = False,
    blocking_size: int = 1024,
    chunk_size_bytes: int = constants.CHUNK_SIZE_BYTES,
) -> Iterator[bytes]:
    """Custom audio iterator for WAV files with configurable bit depth and stereo blocking.
    
    Args:
        audio_files: List of paths to WAV files (provided by calling code)
        num_chunks: Maximum number of chunks to generate
        use_16bit: If True, use 16-bit audio (vocab size 65536), else 8-bit (vocab size 256)
        blocking_size: Size of blocks for stereo processing (in samples)
        chunk_size_bytes: Size of each chunk in bytes
        
    Yields:
        Chunks of audio data as bytes
        
    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If audio files don't exist
    """
    import warnings
    from typing import List
    
    # Validate configuration
    validate_audio_config(use_16bit, blocking_size, chunk_size_bytes)
    
    # Validate audio files
    for file_path in audio_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    logging.info(f"Processing {len(audio_files)} audio files")
    logging.info(f"Configuration: 16-bit={use_16bit}, blocking_size={blocking_size}, "
                f"chunk_size={chunk_size_bytes} bytes")
    
    chunks_generated = 0
    file_index = 0
    
    while chunks_generated < num_chunks and file_index < len(audio_files):
        audio_file = audio_files[file_index]
        
        try:
            # Load audio file
            audio_data, sample_rate, num_channels = load_wav_file(audio_file)
            logging.info(f"Loaded {audio_file}: {num_channels} channels, {sample_rate}Hz, "
                        f"{audio_data.shape[1]} samples")
            
            # Process stereo blocking
            processed_audio = process_stereo_blocking(audio_data, blocking_size)
            
            # Convert to target bit depth
            audio_bytes = convert_to_target_bit_depth(processed_audio, use_16bit)
            
            # Extract chunks
            for chunk in extract_audio_chunks(audio_bytes, chunk_size_bytes):
                if chunks_generated >= num_chunks:
                    break
                    
                yield chunk
                chunks_generated += 1
                
                if chunks_generated % 1000 == 0:
                    logging.info(f"Generated {chunks_generated} chunks")
            
            logging.info(f"Processed {audio_file}: generated chunks from this file")
            
        except Exception as e:
            logging.warning(f"Error processing {audio_file}: {str(e)}")
            # Continue with next file
        
        file_index += 1
    
    logging.info(f"Total chunks generated: {chunks_generated}")
    
    if chunks_generated < num_chunks:
        warnings.warn(
            f"Only generated {chunks_generated} chunks out of requested {num_chunks}. "
            f"Consider adding more audio files.",
            UserWarning
        )
```

#### New Helper Functions:

##### `load_wav_file(file_path: str) -> tuple[np.ndarray, int, int]`
```python
def load_wav_file(file_path: str) -> tuple[np.ndarray, int, int]:
    """Load WAV file and return audio data, sample rate, and number of channels.
    
    Args:
        file_path: Path to the WAV file
        
    Returns:
        Tuple of (audio_data, sample_rate, num_channels)
        audio_data: Shape (channels, samples) for stereo, (1, samples) for mono
        sample_rate: Sample rate in Hz
        num_channels: Number of audio channels (1 for mono, 2 for stereo)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a valid WAV file
    """
    from scipy.io import wavfile
    import os
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    if not file_path.lower().endswith('.wav'):
        raise ValueError(f"File must be a WAV file: {file_path}")
    
    try:
        # Load audio file using scipy.wavfile
        # scipy.wavfile returns (sample_rate, audio_data) where audio_data is (samples, channels)
        sample_rate, audio_data = wavfile.read(file_path)
        
        # Handle mono vs stereo
        if audio_data.ndim == 1:
            # Mono audio
            audio_data = audio_data.reshape(1, -1)  # Shape: (1, samples)
            num_channels = 1
        else:
            # Stereo audio - transpose to (channels, samples)
            audio_data = audio_data.T  # Shape: (channels, samples)
            num_channels = audio_data.shape[0]
        
        # Convert to float32 and normalize to [-1, 1] range
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32767.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483647.0
        elif audio_data.dtype == np.uint8:
            audio_data = (audio_data.astype(np.float32) - 128.0) / 128.0
        else:
            # Assume already in float format
            audio_data = audio_data.astype(np.float32)
        
        return audio_data, sample_rate, num_channels
        
    except Exception as e:
        raise ValueError(f"Error loading WAV file {file_path}: {str(e)}")
```


##### `process_stereo_blocking(audio_data: np.ndarray, blocking_size: int) -> np.ndarray`
```python
def process_stereo_blocking(audio_data: np.ndarray, blocking_size: int) -> np.ndarray:
    """Process stereo audio using interleaved blocking strategy.
    
    Args:
        audio_data: Audio data with shape (channels, samples)
        blocking_size: Size of each block in samples
        
    Returns:
        Processed audio data as 1D array with interleaved blocks
        Pattern: [L_block1, R_block1, L_block2, R_block2, ...]
    """
    channels, total_samples = audio_data.shape
    
    if channels == 1:
        # Mono audio - return as is
        return audio_data[0]  # Return as 1D array
    
    elif channels == 2:
        # Stereo audio - apply blocking strategy
        left_channel = audio_data[0]  # Shape: (samples,)
        right_channel = audio_data[1]  # Shape: (samples,)
        
        # Split channels into blocks
        left_blocks = []
        right_blocks = []
        
        for i in range(0, total_samples, blocking_size):
            left_block = left_channel[i:i + blocking_size]
            right_block = right_channel[i:i + blocking_size]
            
            # Add all blocks, including the final partial block
            left_blocks.append(left_block)
            right_blocks.append(right_block)
        
        # Interleave blocks: [L_block1, R_block1, L_block2, R_block2, ...]
        interleaved_blocks = []
        for left_block, right_block in zip(left_blocks, right_blocks):
            interleaved_blocks.append(left_block)
            interleaved_blocks.append(right_block)
        
        # Concatenate all blocks
        return np.concatenate(interleaved_blocks)
    
    else:
        raise ValueError(f"Unsupported number of channels: {channels}")
```

##### `convert_to_target_bit_depth(audio_data: np.ndarray, use_16bit: bool) -> bytes`
```python
def convert_to_target_bit_depth(audio_data: np.ndarray, use_16bit: bool) -> bytes:
    """Convert audio data to target bit depth and return as bytes.
    
    Args:
        audio_data: Audio data as 1D numpy array (float values in [-1, 1])
        use_16bit: If True, convert to 16-bit, else convert to 8-bit
        
    Returns:
        Audio data as bytes in the target bit depth
    """
    if use_16bit:
        # Convert to 16-bit signed integers
        # Map [-1, 1] to [-32768, 32767]
        audio_16bit = (audio_data * 32767).astype(np.int16)
        return audio_16bit.tobytes()
    else:
        # Convert to 8-bit unsigned integers
        # Map [-1, 1] to [0, 255]
        audio_8bit = ((audio_data + 1.0) * 127.5).astype(np.uint8)
        return audio_8bit.tobytes()
```

##### `extract_audio_chunks(audio_bytes: bytes, chunk_size: int) -> Iterator[bytes]`
```python
def extract_audio_chunks(audio_bytes: bytes, chunk_size: int) -> Iterator[bytes]:
    """Extract fixed-size chunks from audio bytes.
    
    Args:
        audio_bytes: Audio data as bytes
        chunk_size: Size of each chunk in bytes
        
    Yields:
        Chunks of audio data as bytes
    """
    total_bytes = len(audio_bytes)
    
    for i in range(0, total_bytes, chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        
        # Only yield complete chunks
        if len(chunk) == chunk_size:
            yield chunk
        # Skip incomplete chunks at the end
```

##### `validate_blocking_size(blocking_size: int, chunk_size: int) -> None`
```python
def validate_blocking_size(blocking_size: int, chunk_size: int) -> None:
    """Validate blocking size against chunk size and issue warnings if needed.
    
    Args:
        blocking_size: Size of each block in samples
        chunk_size: Size of each chunk in bytes
        
    Raises:
        Warning if blocking_size * 2 < chunk_size
    """
    import warnings
    
    # For stereo audio, each block contributes 2 * blocking_size samples
    # If blocking_size * 2 < chunk_size, chunks won't contain complete L/R block pairs
    if blocking_size * 2 < chunk_size:
        warnings.warn(
            f"Blocking size {blocking_size} is too small for chunk size {chunk_size}. "
            f"Chunks will not contain complete L/R block pairs. "
            f"Consider increasing blocking_size to at least {chunk_size // 2}.",
            UserWarning
        )
```

##### `validate_audio_config(use_16bit: bool, blocking_size: int, chunk_size: int) -> None`
```python
def validate_audio_config(
    use_16bit: bool, 
    blocking_size: int, 
    chunk_size: int
) -> None:
    """Validate audio configuration parameters.
    
    Args:
        use_16bit: Whether to use 16-bit audio
        blocking_size: Size of each block in samples
        chunk_size: Size of each chunk in bytes
        
    Raises:
        ValueError: If any parameter is invalid
    """
    if blocking_size <= 0:
        raise ValueError(f"Blocking size must be positive, got {blocking_size}")
    
    if chunk_size <= 0:
        raise ValueError(f"Chunk size must be positive, got {chunk_size}")
    
    # Validate blocking size against chunk size
    validate_blocking_size(blocking_size, chunk_size)
    
    # Additional validation for 16-bit mode
    if use_16bit:
        # For 16-bit audio, chunk_size should be even (since each sample is 2 bytes)
        if chunk_size % 2 != 0:
            raise ValueError(
                f"For 16-bit audio, chunk_size must be even, got {chunk_size}"
            )
```

#### Modified Constants:
- Add new constants for 16-bit vocabulary size
- Add validation constants for blocking size

### 2. `constants.py` - New Constants

```python
# Audio-specific constants
ALPHABET_SIZE_16BIT = 65536  # For 16-bit audio
MIN_BLOCKING_SIZE = 512      # Minimum blocking size for stereo
DEFAULT_BLOCKING_SIZE = 1024 # Default blocking size
```

### 3. `transformer.py` - Model Architecture Modifications

#### Modified `TransformerConfig`:
```python
@dataclasses.dataclass(kw_only=True)
class TransformerConfig:
    vocab_size: int  # Will be 256 for 8-bit, 65536 for 16-bit
    # ... existing fields
```

#### New Function: `create_audio_transformer_config()`
```python
def create_audio_transformer_config(use_16bit: bool = False) -> TransformerConfig:
    vocab_size = constants.ALPHABET_SIZE_16BIT if use_16bit else constants.ALPHABET_SIZE
    return TransformerConfig(vocab_size=vocab_size, ...)
```

### 4. `train.py` - Training Modifications

#### Modified `train_transformer_decoder()`:
- Add parameters for audio-specific configuration
- Update model initialization to use audio-specific config
- Add validation for blocking size vs chunk size
- **Add wandb logging integration**

#### New Function: `train_audio_transformer()`
```python
def train_audio_transformer(
    audio_files: List[str],
    use_16bit: bool = False,
    blocking_size: int = 1024,
    training_steps: int = 1000,
    log_every: int = 100,
    batch_size: int = 128,
    sequence_length: int = constants.CHUNK_SIZE_BYTES,
    use_tqdm: bool = True,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_config: Optional[dict] = None,
) -> tuple[hk.Params, float]:
```

#### Wandb Logging Integration:
```python
import wandb
from typing import Optional

def train_audio_transformer(
    audio_files: List[str],
    use_16bit: bool = False,
    blocking_size: int = 1024,
    training_steps: int = 1000,
    log_every: int = 100,
    batch_size: int = 128,
    sequence_length: int = constants.CHUNK_SIZE_BYTES,
    use_tqdm: bool = True,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_config: Optional[dict] = None,
) -> tuple[hk.Params, float]:
    """Train audio transformer with wandb logging support.
    
    Args:
        audio_files: List of paths to WAV files (provided by calling code)
        use_16bit: Whether to use 16-bit audio (affects vocabulary size)
        blocking_size: Size of blocks for stereo processing
        training_steps: Number of training steps
        log_every: How often to log metrics
        batch_size: Batch size for training
        sequence_length: Length of sequences in bytes
        use_tqdm: Whether to show progress bar
        wandb_project: Wandb project name (if None, no wandb logging)
        wandb_run_name: Wandb run name
        wandb_config: Additional config to log to wandb
        
    Returns:
        Tuple of (final_params, final_loss)
    """
    
    # Initialize wandb if project is specified
    if wandb_project is not None:
        wandb_config = wandb_config or {}
        wandb_config.update({
            'use_16bit': use_16bit,
            'blocking_size': blocking_size,
            'training_steps': training_steps,
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'num_audio_files': len(audio_files),
        })
        
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=wandb_config
        )
        
        # Log audio file info
        wandb.log({
            'dataset/num_files': len(audio_files),
            'dataset/use_16bit': use_16bit,
            'dataset/blocking_size': blocking_size,
        })
    
    # Validate configuration
    validate_audio_config(use_16bit, blocking_size, sequence_length)
    
    # Create model configuration
    config = create_audio_transformer_config(use_16bit=use_16bit)
    model = hk.transform(
        functools.partial(transformer.transformer_decoder, config=config)
    )
    
    # Create data generator
    data_generator = get_custom_audio_iterator(
        audio_files=audio_files,
        num_chunks=constants.NUM_CHUNKS,
        use_16bit=use_16bit,
        blocking_size=blocking_size,
        chunk_size_bytes=sequence_length,
    )
    dataset = list(data_generator)
    
    # Log dataset statistics
    if wandb_project is not None:
        wandb.log({
            'dataset/total_chunks': len(dataset),
            'dataset/chunk_size_bytes': sequence_length,
            'dataset/vocab_size': config.vocab_size,
        })
    
    def fetch_random_batch() -> np.ndarray:
        batch_list = random.choices(dataset, k=batch_size)
        if use_16bit:
            # For 16-bit, convert bytes to int16 arrays
            batch_list = [np.frombuffer(seq, dtype=np.int16) for seq in batch_list]
        else:
            # For 8-bit, convert bytes to uint8 arrays
            batch_list = [np.frombuffer(seq, dtype=np.uint8) for seq in batch_list]
        return np.array(batch_list, dtype=np.int16 if use_16bit else np.uint8)
    
    # Initialize parameters
    dummy_batch = fetch_random_batch()
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, dummy_batch)
    
    # Log model parameters
    if wandb_project is not None:
        total_params = sum(x.size for x in jax.tree_leaves(params))
        wandb.log({
            'model/total_parameters': total_params,
            'model/vocab_size': config.vocab_size,
            'model/embedding_dim': config.embedding_dim,
            'model/num_layers': config.num_layers,
            'model/num_heads': config.num_heads,
        })
    
    # Make gradient function
    loss_fn = _make_loss_fn(model)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    
    # Make optimizer
    optimizer = optax.adam(learning_rate=1e-4)
    opt_state = optimizer.init(params)
    
    # Log optimizer config
    if wandb_project is not None:
        wandb.log({
            'optimizer/learning_rate': 1e-4,
            'optimizer/optimizer': 'adam',
        })
    
    logging.info('Initialization done, starting training...')
    last_loss = 0.0
    
    for step in tqdm.trange(training_steps, disable=not use_tqdm):
        batch = fetch_random_batch()
        
        params, opt_state, logs = _update_parameters(
            params=params,
            opt_state=opt_state,
            sequences=batch,
            grad_fn=grad_fn,
            optimizer=optimizer,
        )
        
        # Log metrics
        if log_every > 0 and step % log_every == 0:
            logging.info(
                'Step %f, Loss %f, Grad norm %f',
                step,
                logs['loss'],
                logs['grad_norm_unclipped'],
            )
            
            # Log to wandb
            if wandb_project is not None:
                wandb.log({
                    'train/step': step,
                    'train/loss': float(logs['loss']),
                    'train/grad_norm': float(logs['grad_norm_unclipped']),
                    'train/learning_rate': 1e-4,  # Could be made configurable
                })
        
        last_loss = logs['loss']
    
    # Log final metrics
    if wandb_project is not None:
        wandb.log({
            'train/final_loss': float(last_loss),
            'train/total_steps': training_steps,
        })
        
        # Save model parameters as wandb artifact
        np.savez('params.npz', **params)
        artifact = wandb.Artifact('model_params', type='model')
        artifact.add_file('params.npz')
        wandb.log_artifact(artifact)
        
        wandb.finish()
    
    return params, last_loss
```

#### Wandb Configuration and Usage:
```python
# Example usage with wandb logging
def main(_) -> None:
    # Import data_paths helper
    from data_paths import get_train_paths, get_valid_paths
    
    # Get audio file paths using data_paths helper
    audio_directory = "/path/to/audio/directory"
    train_files = get_train_paths(audio_directory)
    val_files = get_valid_paths(audio_directory)
    
    # Train on training files
    params, loss = train_audio_transformer(
        audio_files=train_files,  # Use paths from data_paths
        use_16bit=True,
        blocking_size=1024,
        training_steps=1000,
        log_every=100,
        wandb_project="audio-compression",
        wandb_run_name="16bit-stereo-experiment",
        wandb_config={
            'experiment_type': 'audio_compression',
            'dataset_type': 'custom_wav_directory',
            'audio_directory': audio_directory,
            'num_train_files': len(train_files),
            'num_val_files': len(val_files),
            'notes': 'Testing 16-bit stereo audio compression with directory-based loading',
        }
    )
    
    logging.info('Final loss: %f', loss)
    logging.info('Parameters saved in file params.npz')
```

#### Wandb Dependencies:
```python
# Add to requirements.txt
wandb>=0.15.0
```

### 5. `compressors/language_model.py` - Compression Modifications

#### Modified `_retrieve_predict_fn()`:
- Add support for 16-bit vocabulary
- Update model configuration loading

#### New Function: `compress_audio()`
```python
def compress_audio(
    data: bytes,
    use_16bit: bool = False,
    return_num_padded_bits: bool = False,
) -> bytes | tuple[bytes, int]:
```

## Implementation Details

### 1. WAV File Loading Strategy

#### Directory-Based File Discovery:
```python
# Example usage with data_paths helper
from data_paths import get_train_paths, get_valid_paths

# Get file paths from directory
audio_directory = "/path/to/audio/directory"
train_files = get_train_paths(audio_directory)
val_files = get_valid_paths(audio_directory)

# Use with data loader
data_generator = get_custom_audio_iterator(
    audio_files=train_files,  # List of paths from data_paths
    use_16bit=True,
    blocking_size=1024,
    chunk_size_bytes=constants.CHUNK_SIZE_BYTES,
)
```

#### Data Partitioning:
- **Directory Scanning**: Use `data_paths.py` to discover and partition WAV files
- **Deterministic Splitting**: Consistent train/val splits using random seed
- **File-Level Partitioning**: Split at file level, not chunk level
- **Memory Management**: Load files one at a time to avoid memory issues
- **Progress Tracking**: Track progress across multiple files

### 2. Stereo Blocking Implementation

#### Blocking Pattern:
```
Original Stereo: [L0, L1, L2, ..., R0, R1, R2, ...]
Blocking Size: 3
Result: [L0, L1, L2, R0, R1, R2, L3, L4, L5, R3, R4, R5, ...]
```

#### Implementation Steps:
1. **Load stereo audio** as separate L/R channels
2. **Split channels** into blocks of `blocking_size`
3. **Interleave blocks** following the pattern
4. **Handle remainder** blocks (last block may be smaller)
5. **Convert to bytes** using target bit depth

#### Validation:
```python
def validate_blocking_size(blocking_size: int, chunk_size: int):
    if blocking_size * 2 < chunk_size:
        warnings.warn(
            f"Blocking size {blocking_size} is too small for chunk size {chunk_size}. "
            f"Chunks will not contain complete L/R block pairs."
        )
```

### 3. 16-bit Audio Processing

#### Bit Depth Conversion:
```python
def convert_to_target_bit_depth(audio_data: np.ndarray, use_16bit: bool) -> bytes:
    if use_16bit:
        # Convert to 16-bit signed integers
        audio_16bit = (audio_data * 32767).astype(np.int16)
        return audio_16bit.tobytes()
    else:
        # Convert to 8-bit unsigned (current behavior)
        audio_8bit = ((audio_data + 1.0) * 127.5).astype(np.uint8)
        return audio_8bit.tobytes()
```

#### Model Architecture Impact:
- **Vocabulary Size**: 256 → 65536
- **Embedding Layer**: Larger embedding matrix
- **Output Layer**: Larger output projection
- **Memory Usage**: Significantly increased

### 4. Data Flow Architecture

```
WAV Files (various formats)
    ↓
load_wav_file() → (audio_data, sample_rate, channels)
    ↓
process_stereo_blocking() → (blocked_audio_data)
    ↓
convert_to_target_bit_depth() → (audio_bytes)
    ↓
extract_audio_chunks() → (chunk_iterator)
    ↓
Transformer Training/Compression
```

## Integration Points

### 1. Update `GET_DATA_GENERATOR_FN_DICT`
```python
GET_DATA_GENERATOR_FN_DICT = {
    'enwik9': get_enwik9_iterator,
    'imagenet': get_imagenet_iterator,
    'librispeech': get_librispeech_iterator,
    'custom_audio': get_custom_audio_iterator,  # New entry
    'random': get_random_iterator,
}
```

### 2. Training Integration
```python
# In train.py main function
def main(_) -> None:
    # Import data_paths helper
    from data_paths import get_train_paths, get_valid_paths
    
    # Get audio file paths from directory
    audio_directory = "/path/to/audio/directory"
    train_files = get_train_paths(audio_directory)
    val_files = get_valid_paths(audio_directory)
    
    # Train on training files
    params, loss = train_audio_transformer(
        audio_files=train_files,  # Use paths from data_paths
        use_16bit=True,
        blocking_size=1024,
        training_steps=1000,
        log_every=100,
        wandb_project="audio-compression",
        wandb_run_name="16bit-stereo-experiment",
    )
    
    logging.info('Final loss: %f', loss)
    logging.info('Parameters saved in file params.npz')
```

### 3. Compression Integration
```python
# In compress.py or similar
compressed_data = compress_audio(
    audio_data,
    use_16bit=True,
    return_num_padded_bits=True,
)
```

## Validation and Testing Strategy

### 1. Input Validation
- **File Existence**: Check if WAV files exist
- **File Format**: Validate WAV file headers
- **Audio Properties**: Check sample rate, bit depth, channels
- **Blocking Size**: Validate against chunk size

### 2. Output Validation
- **Chunk Sizes**: Ensure all chunks are correct size
- **Bit Depth**: Verify correct byte representation
- **Stereo Blocking**: Validate blocking pattern
- **Data Integrity**: Check for data corruption

### 3. Performance Testing
- **Memory Usage**: Monitor memory consumption
- **Processing Speed**: Measure processing time
- **Chunk Distribution**: Ensure even distribution across files

## Dependencies and Requirements

### New Dependencies:
```python
# Add to requirements.txt
scipy>=1.7.0
soundfile>=0.10.0
numpy>=1.21.0
```

### Existing Dependencies:
- `audioop` (already used)
- `numpy` (already used)
- `tensorflow_datasets` (for existing loaders)

## Error Handling and Edge Cases

### 1. File Loading Errors
- **Missing Files**: Graceful handling of missing WAV files
- **Corrupted Files**: Skip corrupted files with warning
- **Unsupported Formats**: Error for non-WAV files

### 2. Audio Processing Errors
- **Empty Files**: Skip empty audio files
- **Very Short Files**: Handle files shorter than chunk size
- **Bit Depth Issues**: Handle unsupported audio bit depths

### 3. Stereo Processing Errors
- **Mono Files with Stereo Request**: Convert mono to stereo
- **Stereo Files with Mono Request**: Convert stereo to mono
- **Blocking Size Issues**: Handle remainder blocks

## Configuration Management

### 1. Hyperparameter Validation
```python
def validate_audio_config(
    use_16bit: bool,
    blocking_size: int,
    chunk_size: int,
) -> None:
    # Validate all hyperparameters
    pass
```

### 2. Default Configuration
```python
DEFAULT_AUDIO_CONFIG = {
    'use_16bit': False,
    'blocking_size': 1024,
    'chunk_size_bytes': constants.CHUNK_SIZE_BYTES,
}
```

## Next Steps

1. **Implement core functions** in `data_loaders.py`
2. **Add new constants** to `constants.py`
3. **Update model configuration** in `transformer.py`
4. **Modify training pipeline** in `train.py`
5. **Update compression functions** in `compressors/`
6. **Add comprehensive testing** and validation
7. **Update documentation** and examples

This implementation plan provides a comprehensive roadmap for adapting the existing audio data loader to support your specific requirements while maintaining compatibility with the existing codebase architecture.
