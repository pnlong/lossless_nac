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
DEFAULT_WANDB_PROJECT = "LNAC"
DEFAULT_WANDB_GROUP_NAME = "meow"
DEFAULT_WANDB_RUN_NAME = "audio_model"


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
        default=DEFAULT_WANDB_PROJECT,
        help="Wandb project name for experiment tracking"
    )

    parser.add_argument(
        "--wandb_group_name",
        type=str,
        default=DEFAULT_WANDB_GROUP_NAME,
        help="Wandb group name for experiment tracking"
    )
    
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=DEFAULT_WANDB_RUN_NAME,
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
        
        # Get all files and detect format
        all_files, is_stereo = get_audio_files_with_format(args.audio_dir)
        
        # Split into train/val (using existing logic)
        train_files = get_train_paths(args.audio_dir)
        val_files = get_valid_paths(args.audio_dir)
        
        logging.info(f"Detected audio format: {'stereo' if is_stereo else 'mono'}")
        logging.info(f"Found {len(train_files)} training files and {len(val_files)} validation files")
        if is_stereo:
            logging.info("Stereo files detected - will use stereo blocking strategy")
        else:
            logging.info("Mono files detected - will process as mono")
            
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
        'is_stereo': is_stereo,
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
            wandb_group_name=args.wandb_group_name,
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
