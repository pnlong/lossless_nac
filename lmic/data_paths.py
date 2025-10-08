#!/usr/bin/env python3
"""
Audio Data Paths Helper Script

This script provides helper functions to discover and partition WAV files
for training and validation in the language modeling is compression project.

Usage:
    from data_paths import get_train_paths, get_valid_paths
    
    train_files = get_train_paths("/path/to/audio/directory")
    val_files = get_valid_paths("/path/to/audio/directory")
"""

import os
import random
from typing import List, Optional

# Default split constants
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_RANDOM_SEED = 42


def discover_wav_files(audio_directory: str) -> List[str]:
    """Discover all WAV files in a directory.
    
    Args:
        audio_directory: Path to directory containing WAV files
        
    Returns:
        List of full paths to WAV files
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no WAV files found
    """
    if not os.path.exists(audio_directory):
        raise FileNotFoundError(f"Audio directory not found: {audio_directory}")
    
    if not os.path.isdir(audio_directory):
        raise ValueError(f"Path is not a directory: {audio_directory}")
    
    # Get all files in directory
    all_files = os.listdir(audio_directory)
    
    # Filter for WAV files (case insensitive)
    wav_files = [f for f in all_files if f.lower().endswith('.wav')]
    
    if not wav_files:
        raise ValueError(f"No WAV files found in directory: {audio_directory}")
    
    # Convert to full paths and sort for deterministic behavior
    full_paths = [os.path.join(audio_directory, f) for f in sorted(wav_files)]
    
    return full_paths


def create_train_val_split(
    audio_files: List[str], 
    train_split: float = DEFAULT_TRAIN_SPLIT, 
    val_split: float = DEFAULT_VAL_SPLIT, 
    random_seed: int = DEFAULT_RANDOM_SEED
) -> tuple[List[str], List[str]]:
    """Create deterministic train/validation split from audio files.
    
    Args:
        audio_files: List of paths to audio files
        train_split: Fraction of files for training (default 0.8)
        val_split: Fraction of files for validation (default 0.2)
        random_seed: Random seed for deterministic splitting
        
    Returns:
        Tuple of (train_files, val_files)
        
    Raises:
        ValueError: If splits don't sum to 1.0 or if audio_files is empty
    """
    if not audio_files:
        raise ValueError("audio_files list cannot be empty")
    
    if abs(train_split + val_split - 1.0) > 1e-6:
        raise ValueError(f"train_split + val_split must equal 1.0, got {train_split + val_split}")
    
    if train_split <= 0 or val_split <= 0:
        raise ValueError("train_split and val_split must be positive")
    
    # Create deterministic split
    random.seed(random_seed)
    shuffled_files = audio_files.copy()
    random.shuffle(shuffled_files)
    
    num_train_files = int(len(shuffled_files) * train_split)
    
    train_files = shuffled_files[:num_train_files]
    val_files = shuffled_files[num_train_files:]
    
    return train_files, val_files


def get_train_paths(
    audio_directory: str,
    train_split: float = DEFAULT_TRAIN_SPLIT,
    val_split: float = DEFAULT_VAL_SPLIT,
    random_seed: int = DEFAULT_RANDOM_SEED
) -> List[str]:
    """Get list of WAV file paths for training.
    
    Args:
        audio_directory: Path to directory containing WAV files
        train_split: Fraction of files for training (default 0.8)
        val_split: Fraction of files for validation (default 0.2)
        random_seed: Random seed for deterministic splitting
        
    Returns:
        List of full paths to WAV files for training
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no WAV files found or invalid splits
    """
    # Discover all WAV files
    all_files = discover_wav_files(audio_directory)
    
    # Create train/val split
    train_files, _ = create_train_val_split(all_files, train_split, val_split, random_seed)
    
    return train_files


def get_valid_paths(
    audio_directory: str,
    train_split: float = DEFAULT_TRAIN_SPLIT,
    val_split: float = DEFAULT_VAL_SPLIT,
    random_seed: int = DEFAULT_RANDOM_SEED
) -> List[str]:
    """Get list of WAV file paths for validation.
    
    Args:
        audio_directory: Path to directory containing WAV files
        train_split: Fraction of files for training (default 0.8)
        val_split: Fraction of files for validation (default 0.2)
        random_seed: Random seed for deterministic splitting
        
    Returns:
        List of full paths to WAV files for validation
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no WAV files found or invalid splits
    """
    # Discover all WAV files
    all_files = discover_wav_files(audio_directory)
    
    # Create train/val split
    _, val_files = create_train_val_split(all_files, train_split, val_split, random_seed)
    
    return val_files


def get_all_paths(audio_directory: str) -> List[str]:
    """Get list of all WAV file paths in directory.
    
    Args:
        audio_directory: Path to directory containing WAV files
        
    Returns:
        List of full paths to all WAV files
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no WAV files found
    """
    return discover_wav_files(audio_directory)


def print_dataset_info(audio_directory: str, train_split: float = DEFAULT_TRAIN_SPLIT, val_split: float = DEFAULT_VAL_SPLIT, random_seed: int = DEFAULT_RANDOM_SEED):
    """Print information about the dataset and splits.
    
    Args:
        audio_directory: Path to directory containing WAV files
        train_split: Fraction of files for training
        val_split: Fraction of files for validation
        random_seed: Random seed for deterministic splitting
    """
    try:
        all_files = get_all_paths(audio_directory)
        train_files = get_train_paths(audio_directory, train_split, val_split, random_seed)
        val_files = get_valid_paths(audio_directory, train_split, val_split, random_seed)
        
        print(f"Dataset Information:")
        print(f"  Directory: {audio_directory}")
        print(f"  Total WAV files: {len(all_files)}")
        print(f"  Train files: {len(train_files)} ({len(train_files)/len(all_files)*100:.1f}%)")
        print(f"  Validation files: {len(val_files)} ({len(val_files)/len(all_files)*100:.1f}%)")
        print(f"  Random seed: {random_seed}")
        print(f"  Train split: {train_split}")
        print(f"  Val split: {val_split}")
        
        if len(all_files) <= 10:
            print(f"\nAll files:")
            for i, file_path in enumerate(all_files):
                split_type = "TRAIN" if file_path in train_files else "VAL"
                print(f"  {i+1:2d}. {os.path.basename(file_path)} ({split_type})")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_paths.py <audio_directory> [train_split] [val_split] [random_seed]")
        print("Example: python data_paths.py /path/to/audio 0.8 0.2 42")
        sys.exit(1)
    
    audio_directory = sys.argv[1]
    train_split = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TRAIN_SPLIT
    val_split = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_VAL_SPLIT
    random_seed = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_RANDOM_SEED
    
    print_dataset_info(audio_directory, train_split, val_split, random_seed)


def get_audio_files_with_format(audio_dir: str) -> tuple[List[str], bool]:
    """Get audio files and detect format for proper reconstruction.
    
    Args:
        audio_dir: Path to directory containing WAV files
        
    Returns:
        Tuple of (file_paths, is_stereo)
        is_stereo: True if stereo files detected, False if mono only - important for model reconstruction
    """
    from scipy.io import wavfile
    
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
    
    # Determine format for reconstruction
    is_stereo = len(stereo_files) > 0
    
    # Return all files with detected format
    all_files = stereo_files + mono_files
    return all_files, is_stereo
