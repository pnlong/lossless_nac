# Command Line Integration Plan for Audio Training

## Overview

This document outlines the plan for integrating the custom audio iterator into a command-line interface using argparse. The goal is to create a `train_audio.py` script that allows users to train audio transformers from the command line with configurable parameters.

## Command Line Arguments to Expose

### Required Arguments

1. **`--audio_dir`** (str, required)
   - Path to directory containing WAV files
   - Will use `data_paths.py` to discover and partition files
   - Example: `--audio_dir /path/to/audio/dataset`

### Audio Processing Arguments

2. **`--use_16bit`** (bool, default=False)
   - Whether to use 16-bit audio (vocab size 65536) vs 8-bit (vocab size 256)
   - Flag: if present, use 16-bit; if absent, use 8-bit
   - Example: `--use_16bit` (flag)

3. **`--stereo_blocking_n`** (int, default=1024)
   - Size of blocks for stereo processing (in samples)
   - Must be positive integer
   - Example: `--stereo_blocking_n 2048`

4. **`--chunk_size`** (int, default=2048)
   - Size of each chunk in samples
   - Must be positive integer
   - For 16-bit audio, must be even (since each sample is 2 bytes)
   - Example: `--chunk_size 4096`

### Training Arguments

5. **`--training_steps`** (int, default=1000)
   - Number of training steps
   - Must be positive integer
   - Example: `--training_steps 5000`

6. **`--batch_size`** (int, default=128)
   - Batch size for training
   - Must be positive integer
   - Example: `--batch_size 256`

7. **`--log_every`** (int, default=100)
   - How often to log metrics
   - Must be positive integer
   - Example: `--log_every 50`

8. **`--learning_rate`** (float, default=1e-4)
   - Learning rate for optimizer
   - Must be positive float
   - Example: `--learning_rate 0.0001`

### Model Architecture Arguments

9. **`--embedding_dim`** (int, default=64)
   - Dimension of the first embedding
   - Must be positive integer
   - Example: `--embedding_dim 128`

10. **`--num_layers`** (int, default=4)
    - Number of multi-head attention layers
    - Must be positive integer
    - Example: `--num_layers 6`

11. **`--num_heads`** (int, default=8)
    - Number of heads per layer
    - Must be positive integer
    - Example: `--num_heads 12`

12. **`--widening_factor`** (int, default=4)
    - How much larger the hidden layer should be compared to embedding_dim
    - Must be positive integer
    - Example: `--widening_factor 2`

### Wandb Logging Arguments

13. **`--wandb_project`** (str, optional)
    - Wandb project name for experiment tracking
    - If not provided, no wandb logging
    - Example: `--wandb_project "audio-compression"`

14. **`--wandb_run_name`** (str, optional)
    - Wandb run name
    - If not provided, auto-generate based on parameters
    - Example: `--wandb_run_name "16bit-stereo-experiment"`

### Output Arguments

15. **`--output_dir`** (str, default="./")
    - Directory to save model parameters
    - Will create directory if it doesn't exist
    - Example: `--output_dir ./models/`

16. **`--model_name`** (str, default="audio_model")
    - Base name for saved model files
    - Will save as `{model_name}.npz`
    - Example: `--model_name "my_audio_model"`

### Utility Arguments

17. **`--seed`** (int, default=42)
    - Random seed for reproducibility
    - Must be non-negative integer
    - Example: `--seed 123`

18. **`--num_chunks`** (int, default=488281)
    - Maximum number of chunks to generate from audio files
    - Must be positive integer
    - Example: `--num_chunks 100000`


## Implementation Plan

### 1. Script Structure (`train_audio.py`)

```python
#!/usr/bin/env python3
"""
Command-line interface for training audio transformers.

Usage:
    python train_audio.py --audio_dir /path/to/audio --use_16bit --wandb_project "audio-compression"
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the language_modeling_is_compression module to path
sys.path.append(str(Path(__file__).parent / "language_modeling_is_compression"))

from language_modeling_is_compression import train
from data_paths import get_train_paths, get_valid_paths

# Default values as constants
DEFAULT_STEREO_BLOCKING_N = 1024
DEFAULT_CHUNK_SIZE = 2048
DEFAULT_TRAINING_STEPS = 1000
DEFAULT_BATCH_SIZE = 128
DEFAULT_LOG_EVERY = 100
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_NUM_LAYERS = 4
DEFAULT_NUM_HEADS = 8
DEFAULT_WIDENING_FACTOR = 4
DEFAULT_OUTPUT_DIR = "./"
DEFAULT_MODEL_NAME = "audio_model"
DEFAULT_SEED = 42
DEFAULT_NUM_CHUNKS = 488281

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train audio transformer with custom audio data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Path to directory containing WAV files"
    )
    
    # Audio processing arguments
    parser.add_argument(
        "--use_16bit",
        action="store_true",
        help="Use 16-bit audio (vocab size 65536) instead of 8-bit (vocab size 256)"
    )
    
    parser.add_argument(
        "--stereo_blocking_n",
        type=int,
        default=DEFAULT_STEREO_BLOCKING_N,
        help="Size of blocks for stereo processing (in samples)"
    )
    
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Size of each chunk in samples"
    )
    
    # Training arguments
    parser.add_argument(
        "--training_steps",
        type=int,
        default=DEFAULT_TRAINING_STEPS,
        help="Number of training steps"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--log_every",
        type=int,
        default=DEFAULT_LOG_EVERY,
        help="How often to log metrics"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for optimizer"
    )
    
    # Model architecture arguments
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help="Dimension of the first embedding"
    )
    
    parser.add_argument(
        "--num_layers",
        type=int,
        default=DEFAULT_NUM_LAYERS,
        help="Number of multi-head attention layers"
    )
    
    parser.add_argument(
        "--num_heads",
        type=int,
        default=DEFAULT_NUM_HEADS,
        help="Number of heads per layer"
    )
    
    parser.add_argument(
        "--widening_factor",
        type=int,
        default=DEFAULT_WIDENING_FACTOR,
        help="How much larger the hidden layer should be compared to embedding_dim"
    )
    
    # Wandb logging arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        help="Wandb project name for experiment tracking"
    )
    
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        help="Wandb run name (auto-generated if not provided)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save model parameters"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Base name for saved model files"
    )
    
    # Utility arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=DEFAULT_NUM_CHUNKS,
        help="Maximum number of chunks to generate from audio files"
    )
    
    
    return parser.parse_args()

def validate_arguments(args):
    """Validate command line arguments."""
    # Check if audio directory exists
    if not os.path.exists(args.audio_dir):
        raise ValueError(f"Audio directory does not exist: {args.audio_dir}")
    
    # Check if audio directory contains WAV files
    wav_files = list(Path(args.audio_dir).glob("*.wav"))
    if not wav_files:
        raise ValueError(f"No WAV files found in directory: {args.audio_dir}")
    
    # Validate positive integers
    positive_int_args = [
        "stereo_blocking_n", "chunk_size", "training_steps", "batch_size",
        "log_every", "embedding_dim", "num_layers", "num_heads", "widening_factor",
        "num_chunks"
    ]
    
    for arg_name in positive_int_args:
        value = getattr(args, arg_name)
        if value <= 0:
            raise ValueError(f"{arg_name} must be positive, got {value}")
    
    # Validate learning rate
    if args.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {args.learning_rate}")
    
    # Validate seed
    if args.seed < 0:
        raise ValueError(f"seed must be non-negative, got {args.seed}")
    
    # Validate 16-bit chunk size (must be even since each sample is 2 bytes)
    if args.use_16bit and args.chunk_size % 2 != 0:
        raise ValueError(f"chunk_size must be even for 16-bit audio, got {args.chunk_size}")
    
    # Validate blocking size vs chunk size
    if args.stereo_blocking_n * 2 < args.chunk_size:
        logging.warning(
            f"Blocking size {args.stereo_blocking_n} is too small for chunk size {args.chunk_size}. "
            f"Chunks will not contain complete L/R block pairs."
        )
    

def setup_logging(args):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{args.model_name}_training.log")
        ]
    )

def generate_wandb_run_name(args):
    """Generate wandb run name based on arguments."""
    if args.wandb_run_name:
        return args.wandb_run_name
    
    bit_depth = "16bit" if args.use_16bit else "8bit"
    return f"{bit_depth}_block{args.stereo_blocking_n}_chunk{args.chunk_size}_steps{args.training_steps}"

def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Validate arguments
    validate_arguments(args)
    
    # Setup logging
    setup_logging(args)
    
    # Log configuration
    logging.info("Starting audio transformer training")
    logging.info(f"Configuration: {vars(args)}")
    
    # Set random seed
    import random
    import numpy as np
    import jax
    random.seed(args.seed)
    np.random.seed(args.seed)
    jax.random.PRNGKey(args.seed)
    
    # Get audio file paths with format detection
    try:
        from data_paths import get_audio_files_with_format
        
        # Get all files (always processed as mono)
        all_files, detected_format = get_audio_files_with_format(args.audio_dir)
        
        # Split into train/val (using existing logic)
        train_files = get_train_paths(args.audio_dir)
        val_files = get_valid_paths(args.audio_dir)
        
        logging.info(f"Audio processing format: {detected_format}")
        logging.info(f"Found {len(train_files)} training files and {len(val_files)} validation files")
        logging.info("All audio files will be processed as mono (stereo files converted to mono)")
            
    except Exception as e:
        raise ValueError(f"Error getting audio file paths: {e}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate wandb run name
    wandb_run_name = generate_wandb_run_name(args)
    
    # Prepare wandb config
    wandb_config = {
        'experiment_type': 'audio_compression',
        'dataset_type': 'custom_wav_directory',
        'audio_directory': args.audio_dir,
        'num_train_files': len(train_files),
        'num_val_files': len(val_files),
        'seed': args.seed,
        'learning_rate': args.learning_rate,
        'embedding_dim': args.embedding_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'widening_factor': args.widening_factor,
    }
    
    # Train model
    try:
        params, loss = train.train_audio_transformer(
            audio_files=train_files,
            use_16bit=args.use_16bit,
            blocking_size=args.stereo_blocking_n,
            training_steps=args.training_steps,
            log_every=args.log_every,
            batch_size=args.batch_size,
            sequence_length=args.chunk_size,
            use_tqdm=True,
            wandb_project=args.wandb_project,
            wandb_run_name=wandb_run_name,
            wandb_config=wandb_config,
        )
        
        # Save model
        model_path = os.path.join(args.output_dir, f"{args.model_name}.npz")
        import numpy as np
        np.savez(model_path, **params)
        
        logging.info(f"Training completed successfully!")
        logging.info(f"Final loss: {loss}")
        logging.info(f"Model saved to: {model_path}")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
```

### 2. Key Features

1. **Comprehensive Argument Parsing**: All relevant parameters exposed with sensible defaults
2. **Argument Validation**: Extensive validation with helpful error messages
3. **Logging Setup**: Both console and file logging
4. **Wandb Integration**: Automatic run name generation and config logging
5. **Error Handling**: Graceful error handling with informative messages
6. **Reproducibility**: Random seed setting for consistent results
7. **File Discovery**: Automatic WAV file discovery using `data_paths.py`
8. **Model Saving**: Automatic model parameter saving with configurable paths

### 3. Usage Examples

```bash
# Basic 8-bit training
python train_audio.py --audio_dir /path/to/audio

# 16-bit stereo training with wandb
python train_audio.py --audio_dir /path/to/audio --use_16bit --wandb_project "audio-compression"

# Custom model architecture
python train_audio.py --audio_dir /path/to/audio --embedding_dim 128 --num_layers 6 --num_heads 12

# High-performance training
python train_audio.py --audio_dir /path/to/audio --batch_size 256 --training_steps 5000 --learning_rate 0.0002

# Custom blocking and chunking
python train_audio.py --audio_dir /path/to/audio --stereo_blocking_n 2048 --chunk_size 4096

# Save to custom location
python train_audio.py --audio_dir /path/to/audio --output_dir ./models/ --model_name "my_experiment"
```

### 4. Integration Points

1. **Import Path**: Add `language_modeling_is_compression` to Python path
2. **Data Paths**: Use existing `data_paths.py` for file discovery with auto-detection
3. **Training Function**: Call `train.train_audio_transformer()` with parsed arguments
4. **Model Saving**: Use numpy to save parameters in NPZ format
5. **Logging**: Integrate with existing logging infrastructure

### 5. Required Updates to `data_paths.py`

Add stereo/mono auto-detection functionality:

```python
def detect_audio_format(audio_dir: str) -> dict:
    """Detect audio format (stereo/mono) from WAV files in directory.
    
    Args:
        audio_dir: Path to directory containing WAV files
        
    Returns:
        Dictionary with format information:
        {
            'has_stereo': bool,
            'has_mono': bool,
            'stereo_files': List[str],
            'mono_files': List[str],
            'recommended_format': str  # 'stereo', 'mono', or 'mixed'
        }
    """
    from scipy.io import wavfile
    import os
    
    stereo_files = []
    mono_files = []
    
    for file_path in Path(audio_dir).glob("*.wav"):
        try:
            _, audio_data = wavfile.read(str(file_path))
            if audio_data.ndim == 1:
                mono_files.append(str(file_path))
            else:
                stereo_files.append(str(file_path))
        except Exception as e:
            logging.warning(f"Could not read {file_path}: {e}")
    
    has_stereo = len(stereo_files) > 0
    has_mono = len(mono_files) > 0
    
    if has_stereo and has_mono:
        recommended_format = "mixed"
    elif has_stereo:
        recommended_format = "stereo"
    else:
        recommended_format = "mono"
    
    return {
        'has_stereo': has_stereo,
        'has_mono': has_mono,
        'stereo_files': stereo_files,
        'mono_files': mono_files,
        'recommended_format': recommended_format
    }

def get_audio_files_with_format(audio_dir: str) -> tuple[List[str], str]:
    """Get audio files and detected format.
    
    Args:
        audio_dir: Path to directory containing WAV files
        
    Returns:
        Tuple of (file_paths, format_type)
        format_type: Always 'mono' (stereo files are converted to mono)
    """
    format_info = detect_audio_format(audio_dir)
    
    # Always return all files and process as mono
    all_files = format_info['stereo_files'] + format_info['mono_files']
    return all_files, 'mono'
```

### 5. Error Handling Strategy

1. **Argument Validation**: Validate all arguments before starting training
2. **File System Checks**: Verify audio directory and WAV files exist
3. **Training Errors**: Catch and log training failures gracefully
4. **Wandb Errors**: Handle wandb initialization failures
5. **Model Saving Errors**: Handle file system errors during model saving

This implementation provides a complete command-line interface for training audio transformers with full configurability and robust error handling.
