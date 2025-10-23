#!/usr/bin/env python3
"""
Zero-Shot Language Model Evaluation Script for LIBRISPEECH Dataset

This script evaluates pre-trained language models on the LIBRISPEECH dataset without training.
It provides comprehensive configurability for audio processing parameters and supports
multiple evaluation scenarios including baseline comparisons.

Usage:
    python zero_shot_librispeech.py --model_path /path/to/model.npz
    python zero_shot_librispeech.py --model_path /path/to/model.npz --librispeech_split train-clean-100 --bit_depth 8 --gpu --verbose
"""

import os
# Set JAX platform to CPU before any other imports to avoid conflicts with PyTorch
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator, Callable
import warnings

# Add the language_modeling_is_compression module to path
sys.path.append(str(Path(__file__).parent / "language_modeling_is_compression"))

import numpy as np
import haiku as hk
from absl import logging as absl_logging

# Import torch for Llama models (lazy loaded to avoid conflicts)
try:
    import torch
    import torchaudio
    import torchaudio.datasets
except ImportError:
    torch = None
    torchaudio = None

# Import framework components first
from language_modeling_is_compression import constants
from language_modeling_is_compression import data_loaders
from language_modeling_is_compression import utils
from language_modeling_is_compression import transformer
from language_modeling_is_compression.compressors import compressor
from language_modeling_is_compression.compress import evaluate_compressor_chunked, evaluate_compressor_unchunked

# Llama imports will be done lazily to avoid conflicts with Haiku/JAX
LLAMA_AVAILABLE = None  # Will be set when actually needed


def _import_llama_modules():
    """Lazily import Llama modules to avoid conflicts with Haiku/JAX."""
    global LLAMA_AVAILABLE
    
    if LLAMA_AVAILABLE is not None:
        return LLAMA_AVAILABLE
    
    try:
        import torch
        import transformers
        from transformers import LlamaForCausalLM, LlamaTokenizer
        LLAMA_AVAILABLE = True
        return True
    except ImportError as e:
        LLAMA_AVAILABLE = False
        logging.warning(f"Llama transformers not available: {e}")
        return False
    except Exception as e:
        LLAMA_AVAILABLE = False
        logging.warning(f"Unexpected error importing Llama modules: {e}")
        return False


# Default values as constants
DEFAULT_STEREO_BLOCKING_N = 1024
DEFAULT_CHUNK_SIZE = 2048  # samples per channel
DEFAULT_NUM_CHUNKS = 1000
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_NUM_LAYERS = 4
DEFAULT_NUM_HEADS = 8
DEFAULT_WIDENING_FACTOR = 4
DEFAULT_BASELINE_COMPRESSORS = ['gzip', 'flac', 'lzma']
DEFAULT_VERBOSE = False
DEFAULT_SLOW_COMPRESSION = False
DEFAULT_MAX_LENGTH = 16384
DEFAULT_BATCH_SIZE = 32
DEFAULT_CHUNKS_PER_FILE = 50
DEFAULT_LIBRISPEECH_SPLIT = 'dev-clean'
DEFAULT_MAX_FILES = None


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Suppress absl logging if not verbose
    if not verbose:
        absl_logging.set_verbosity(absl_logging.ERROR)


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation of language models on LIBRISPEECH dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model file (.npz) or Llama model directory (e.g., './llama-2-7b-chat-hf')"
    )
    
    # LIBRISPEECH dataset parameters
    parser.add_argument(
        "--librispeech_split",
        type=str,
        default=DEFAULT_LIBRISPEECH_SPLIT,
        help="LIBRISPEECH dataset split to use (e.g., 'train-clean-100', 'dev-clean', 'test-clean')"
    )
    
    parser.add_argument(
        "--max_files",
        type=int,
        default=DEFAULT_MAX_FILES,
        help="Maximum number of files to process (None for all files)"
    )
    
    # Audio processing parameters
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Size of each audio chunk in samples"
    )
    
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=DEFAULT_NUM_CHUNKS,
        help="Maximum number of chunks to evaluate"
    )
    
    # Model configuration parameters
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help="Embedding dimension for model"
    )
    
    parser.add_argument(
        "--num_layers",
        type=int,
        default=DEFAULT_NUM_LAYERS,
        help="Number of transformer layers"
    )
    
    parser.add_argument(
        "--num_heads",
        type=int,
        default=DEFAULT_NUM_HEADS,
        help="Number of attention heads"
    )
    
    parser.add_argument(
        "--widening_factor",
        type=int,
        default=DEFAULT_WIDENING_FACTOR,
        help="Feedforward network scaling factor"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--compressor",
        type=str,
        default="language_model",
        choices=list(compressor.COMPRESS_FN_DICT.keys()),
        help="Compressor to use for comparison"
    )
    
    parser.add_argument(
        "--baseline_compressors",
        type=str,
        nargs="*",
        default=DEFAULT_BASELINE_COMPRESSORS,
        choices=list(compressor.COMPRESS_FN_DICT.keys()),
        help="List of baseline compressors to compare against"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save detailed results (optional)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=DEFAULT_VERBOSE,
        help="Enable verbose logging"
    )
    
    # Advanced options
    parser.add_argument(
        "--slow_compression",
        action="store_true",
        default=DEFAULT_SLOW_COMPRESSION,
        help="Use slow lossless compression mode"
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration if available"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum sequence length for Llama models"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for Llama model inference"
    )
    
    parser.add_argument(
        "--chunks_per_file",
        type=int,
        default=DEFAULT_CHUNKS_PER_FILE,
        help="Maximum number of chunks to sample from each audio file"
    )
    
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for file sampling (for reproducible results)"
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    # Check if model file exists (only for framework models)
    model_type = detect_model_type(args.model_path)
    if model_type == "framework":
        if not os.path.exists(args.model_path):
            raise ValueError(f"Model file does not exist: {args.model_path}")
    
    # Validate positive integers
    positive_int_args = [
        "chunk_size", "num_chunks", "embedding_dim",
        "num_layers", "num_heads", "widening_factor", "max_length", "batch_size", "chunks_per_file"
    ]
    
    for arg_name in positive_int_args:
        value = getattr(args, arg_name)
        if value <= 0:
            raise ValueError(f"{arg_name} must be positive, got {value}")
    
    # Validate max_files if provided
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError(f"max_files must be positive, got {args.max_files}")
    
    # Validate baseline compressors
    if args.baseline_compressors:
        invalid_compressors = [c for c in args.baseline_compressors if c not in compressor.COMPRESS_FN_DICT]
        if invalid_compressors:
            raise ValueError(f"Invalid baseline compressors: {invalid_compressors}")
    
    # Validate LIBRISPEECH split
    valid_splits = [
        'train-clean-100', 'train-clean-360', 'train-other-500',
        'dev-clean', 'dev-other', 'test-clean', 'test-other'
    ]
    if args.librispeech_split not in valid_splits:
        raise ValueError(f"Invalid LIBRISPEECH split: {args.librispeech_split}. Valid splits: {valid_splits}")


def detect_model_type(model_path: str) -> str:
    """Detect whether model is Llama or framework format."""
    if model_path.startswith("llama:"):
        return "llama"
    elif model_path.endswith(".npz"):
        return "framework"
    elif os.path.exists(model_path) and os.path.isdir(model_path):
        # Check for Hugging Face format (config.json + pytorch_model.bin)
        config_file = os.path.join(model_path, "config.json")
        pytorch_model_file = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(config_file) and os.path.exists(pytorch_model_file):
            try:
                import json
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    if config.get("model_type") == "llama":
                        return "llama"
            except Exception as e:
                logging.debug(f"Error reading config.json: {e}")
        
        # Check for original Llama format (consolidated.00.pth + params.json)
        consolidated_file = os.path.join(model_path, "consolidated.00.pth")
        params_file = os.path.join(model_path, "params.json")
        if os.path.exists(consolidated_file) and os.path.exists(params_file):
            return "llama"
        
        # Check for other Llama model indicators
        bin_files = [f for f in os.listdir(model_path) if f.endswith('.bin')]
        pth_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
        if bin_files or pth_files:
            return "llama"
            
        # If directory contains "llama" in the name, assume it's a Llama model
        if "llama" in model_path.lower():
            return "llama"
            
    elif "/" in model_path and not os.path.exists(model_path):
        # Likely a Hugging Face model name (contains slash but file doesn't exist)
        return "llama"
    else:
        raise ValueError(f"Unknown model format: {model_path}")


def get_librispeech_files(split: str, max_files: Optional[int] = None, random_seed: int = 42) -> List[Tuple[str, int]]:
    """Get LIBRISPEECH audio files and their sample rates with random sampling."""
    if torchaudio is None:
        raise ImportError("torchaudio not available. Install with: pip install torchaudio")
    
    try:
        # Create LIBRISPEECH dataset
        librispeech_dataset = torchaudio.datasets.LIBRISPEECH(
            root="/graft3/datasets/pnlong/lnac/sashimi/data/librispeech",
            url=split,
            download=True
        )
        
        total_files = len(librispeech_dataset)
        # logging.info(f"LIBRISPEECH {split} dataset has {total_files} files")
        
        # Determine how many files to sample
        if max_files is None:
            files_to_sample = total_files
        else:
            files_to_sample = min(max_files, total_files)
        
        # Generate random indices for sampling
        if files_to_sample < total_files:
            # Random sampling
            np.random.seed(random_seed)
            indices = np.random.choice(total_files, size=files_to_sample, replace=False)
            indices = sorted(indices)  # Sort for deterministic order
            # logging.info(f"Randomly sampling {files_to_sample} files from {total_files} total files")
        else:
            # Use all files
            indices = list(range(total_files))
            # logging.info(f"Using all {total_files} files")
        
        # Extract only the sampled files
        audio_files = []
        for i, idx in enumerate(indices):
            waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = librispeech_dataset[idx]
            
            # Create a file identifier
            file_id = f"{speaker_id}-{chapter_id}-{utterance_id}"
            audio_files.append((file_id, sample_rate, waveform))
            
            if i % 100 == 0:
                logging.debug(f"Processed {i+1}/{len(indices)} sampled files")
        
        # logging.info(f"Extracted {len(audio_files)} audio files from LIBRISPEECH {split}")
        return audio_files
        
    except Exception as e:
        raise ValueError(f"Error loading LIBRISPEECH dataset: {str(e)}")


def detect_bit_depth_from_librispeech(audio_files: List[Tuple[str, int, Any]]) -> int:
    """Detect bit depth from LIBRISPEECH audio data."""
    if not audio_files:
        logging.warning("No audio files found, defaulting to 8-bit")
        return 8
    
    # Check first few files
    for i, (file_id, sample_rate, waveform) in enumerate(audio_files[:3]):
        try:
            # Convert tensor to numpy
            audio_data = waveform.numpy()
            
            # Detect bit depth from data type and values
            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                # Check if values are in [-1, 1] range (normalized)
                if np.all(audio_data >= -1.0) and np.all(audio_data <= 1.0):
                    # This is normalized audio from LIBRISPEECH
                    logging.info(f"Audio file {file_id} is normalized float32/64 - will convert to 8-bit PCM")
                    return 8  # We'll convert this to 8-bit PCM
                else:
                    logging.warning(f"Audio file {file_id} has unexpected float range")
                    return 8
            elif audio_data.dtype == np.int16:
                logging.warning(f"Audio file {file_id} is 16-bit (int16), but expecting 8-bit")
                return 16  # Report actual bit depth
            elif audio_data.dtype == np.int8 or audio_data.dtype == np.uint8:
                logging.info(f"Audio file {file_id} is 8-bit as expected")
                return 8
            else:
                logging.warning(f"Audio file {file_id} has unexpected dtype: {audio_data.dtype}")
                return 8
                
        except Exception as e:
            logging.warning(f"Error checking bit depth for {file_id}: {e}")
            continue
    
    # Fallback to 8-bit if all checks fail
    logging.warning("Could not detect bit depth from LIBRISPEECH files, defaulting to 8-bit")
    return 8


def detect_stereo_audio_librispeech(audio_files: List[Tuple[str, int, Any]]) -> bool:
    """Detect if any LIBRISPEECH audio files are stereo."""
    # LIBRISPEECH is mono, so this always returns False
    return False


def create_librispeech_raw_audio_generator(args: argparse.Namespace, bit_depth: int) -> Iterator[bytes]:
    """Create generator that yields raw audio chunks from LIBRISPEECH dataset.
    
    This generator provides raw audio data in the original format (like zero_shot.py).
    The language model will process this raw data internally.
    """
    from language_modeling_is_compression import ascii_mapping
    
    # Get LIBRISPEECH audio files
    try:
        audio_files = get_librispeech_files(args.librispeech_split, args.max_files, args.random_seed)
        if not audio_files:
            raise ValueError(f"No files found in LIBRISPEECH {args.librispeech_split}")
    except Exception as e:
        raise ValueError(f"Error loading LIBRISPEECH files: {str(e)}")
    
    # logging.info(f"Creating LIBRISPEECH raw audio generator with {len(audio_files)} files")
    
    # DEBUG: Log first file info
    if audio_files:
        first_file_id, first_sample_rate, first_waveform = audio_files[0]
        first_audio_data = first_waveform.numpy()
        logging.debug(f"First file: {first_file_id}, shape: {first_audio_data.shape}, dtype: {first_audio_data.dtype}, size: {first_audio_data.size}")
    
    # chunk_size is samples for mono audio
    samples_per_chunk = args.chunk_size
    bytes_per_sample = ascii_mapping.calculate_bytes_per_sample(bit_depth)
    chunk_size_bytes = samples_per_chunk * bytes_per_sample
    
    # DEBUG: Log chunking parameters
    logging.debug(f"LIBRISPEECH raw audio generator - chunk_size: {args.chunk_size} samples (mono)")
    logging.debug(f"LIBRISPEECH raw audio generator - bit_depth: {bit_depth}")
    logging.debug(f"LIBRISPEECH raw audio generator - bytes_per_sample: {bytes_per_sample}")
    logging.debug(f"LIBRISPEECH raw audio generator - samples_per_chunk: {samples_per_chunk}")
    logging.debug(f"LIBRISPEECH raw audio generator - chunk_size_bytes: {chunk_size_bytes}")
    
    chunk_count = 0
    files_processed = 0
    
    for file_id, sample_rate, waveform in audio_files:
        if chunk_count >= args.num_chunks:
            break
            
        try:
            # Convert tensor to numpy
            audio_data = waveform.numpy()
            
            # DEBUG: Check for empty data
            if audio_data.size == 0:
                logging.warning(f"LIBRISPEECH generator - File: {file_id}, EMPTY AUDIO DATA!")
                continue
            
            # LIBRISPEECH is mono, so we expect 1D array
            if len(audio_data.shape) > 1:
                # If somehow stereo, take first channel
                audio_data = audio_data[0] if audio_data.shape[0] > 1 else audio_data.flatten()
                logging.debug(f"LIBRISPEECH generator - File: {file_id}, Converted to mono: {audio_data.shape}")
            else:
                # Mono audio - keep as is
                audio_data = audio_data
                logging.debug(f"LIBRISPEECH generator - File: {file_id}, Mono shape: {audio_data.shape}")
            
            # Convert to target bit depth format (like zero_shot.py raw generator)
            if bit_depth == 8:
                # Convert normalized float to 8-bit PCM [-128, 127] (like zero_shot.py)
                if audio_data.dtype in [np.float32, np.float64]:
                    # Convert from [-1, 1] to [-128, 127] (signed 8-bit)
                    audio_data = (audio_data * 127.0).astype(np.int8)
                else:
                    audio_data = audio_data.astype(np.int8)
            elif bit_depth == 16:
                # Convert normalized float to 16-bit
                if audio_data.dtype in [np.float32, np.float64]:
                    # Convert from [-1, 1] to [-32768, 32767]
                    audio_data = (audio_data * 32767.0).astype(np.int16)
                else:
                    audio_data = audio_data.astype(np.int16)
            
            # Ensure we have complete sample chunks
            total_samples = audio_data.size
            complete_chunks = (total_samples // samples_per_chunk) * samples_per_chunk
            
            # DEBUG: Log chunking info
            logging.debug(f"LIBRISPEECH generator - Total samples: {total_samples}")
            logging.debug(f"LIBRISPEECH generator - Complete chunks: {complete_chunks}")
            logging.debug(f"LIBRISPEECH generator - Samples per chunk: {samples_per_chunk}")
            
            # Truncate to complete chunks only
            audio_data = audio_data[:complete_chunks]
            
            # Chunk by sample count (not byte count)
            for i in range(0, len(audio_data), samples_per_chunk):
                if chunk_count >= args.num_chunks:
                    break
                    
                sample_chunk = audio_data[i:i + samples_per_chunk]
                if len(sample_chunk) == samples_per_chunk:
                    # Convert sample chunk to bytes preserving format
                    chunk_bytes = sample_chunk.tobytes()
                    
                    # DEBUG: Log chunk details
                    logging.debug(f"LIBRISPEECH generator - Chunk #{chunk_count + 1}")
                    logging.debug(f"LIBRISPEECH generator - Sample chunk shape: {sample_chunk.shape}")
                    logging.debug(f"LIBRISPEECH generator - Sample chunk size: {sample_chunk.size} samples")
                    logging.debug(f"LIBRISPEECH generator - Chunk bytes: {len(chunk_bytes)} bytes")
                    logging.debug(f"LIBRISPEECH generator - Expected bytes: {samples_per_chunk * bytes_per_sample} bytes")
                    
                    chunk_count += 1
                    logging.debug(f"LIBRISPEECH generator - Yielding chunk #{chunk_count}, {len(chunk_bytes)} bytes")
                    yield chunk_bytes
            
            files_processed += 1
            
        except Exception as e:
            logging.warning(f"Error processing LIBRISPEECH file {file_id}: {e}")
            continue
    
    logging.debug(f"LIBRISPEECH raw audio generator completed: {chunk_count} chunks from {files_processed} files")


def setup_librispeech_audio_data_generator(args: argparse.Namespace, bit_depth: int) -> Iterator[bytes]:
    """Set up LIBRISPEECH audio data generator with bit depth support."""
    # Get LIBRISPEECH audio files
    try:
        audio_files = get_librispeech_files(args.librispeech_split, args.max_files, args.random_seed)
        if not audio_files:
            raise ValueError(f"No files found in LIBRISPEECH {args.librispeech_split}")
    except Exception as e:
        raise ValueError(f"Error loading LIBRISPEECH files: {str(e)}")
    
    # LIBRISPEECH is mono audio, so no stereo processing needed
    logging.info("LIBRISPEECH is mono audio - using standard processing")
    effective_max_length = args.max_length
    
    # Store the effective max_length for use in Llama models
    args.effective_max_length = effective_max_length
    
    # Create data generator with bit depth support
    from language_modeling_is_compression import audio_processing_extended
    from language_modeling_is_compression import ascii_mapping
    logging.info("=== CREATING LIBRISPEECH DATA GENERATOR ===")
    logging.info(f"Using LIBRISPEECH dataset: {args.librispeech_split}")
    logging.info(f"Parameters: bit_depth={bit_depth}, chunk_size={args.chunk_size} (mono audio)")
    
    # Calculate chunk size in bytes for the extended iterator
    bytes_per_sample = ascii_mapping.calculate_bytes_per_sample(bit_depth)
    chunk_size_bytes = args.chunk_size * bytes_per_sample
    
    # Create a custom iterator for LIBRISPEECH data
    def librispeech_data_generator():
        chunk_count = 0
        files_processed = 0
        
        for file_id, sample_rate, waveform in audio_files:
            if chunk_count >= args.num_chunks:
                break
                
            try:
                # Convert tensor to numpy
                audio_data = waveform.numpy()
                
                # LIBRISPEECH is mono, so we expect 1D array
                if len(audio_data.shape) > 1:
                    # If somehow stereo, take first channel
                    audio_data = audio_data[0] if audio_data.shape[0] > 1 else audio_data.flatten()
                else:
                    # Mono audio - keep as is
                    audio_data = audio_data
                
                # Normalize to [-1, 1] if needed
                if audio_data.dtype in [np.int16, np.int32]:
                    if audio_data.dtype == np.int16:
                        audio_data = audio_data.astype(np.float32) / 32767.0
                    elif audio_data.dtype == np.int32:
                        audio_data = audio_data.astype(np.float32) / 2147483647.0
                elif audio_data.dtype == np.uint8:
                    audio_data = (audio_data.astype(np.float32) - 128.0) / 128.0
                
                # Convert to target bit depth following paper's approach
                if bit_depth == 8:
                    # Paper approach: Convert int16 to uint8 [0, 255] directly
                    if audio_data.dtype in [np.float32, np.float64]:
                        # LIBRISPEECH provides normalized floats, convert back to int16 range first
                        # Convert from [-1, 1] to int16 range [-32768, 32767]
                        audio_data = (audio_data * 32767.0).astype(np.int16)
                    
                    # Now convert int16 to uint8 [0, 255] as per paper
                    # Map int16 [-32768, 32767] to uint8 [0, 255]
                    # Convert to float first to avoid overflow, then to uint8
                    audio_data = ((audio_data.astype(np.float32) + 32768) / 256).astype(np.uint8)
                    processed_bytes = audio_data.tobytes()
                else:
                    # Use the existing function for other bit depths
                    processed_bytes = audio_processing_extended.convert_to_target_bit_depth_extended(audio_data, bit_depth)
                
                # Apply ASCII mapping
                ascii_mapping_fn = ascii_mapping.get_ascii_mapping_function_for_bit_depth(bit_depth)
                ascii_data, dropped_lsb_bits = ascii_mapping_fn(processed_bytes)
                
                # Yield the ASCII data
                chunk_count += 1
                files_processed += 1
                logging.debug(f"LIBRISPEECH data generator yielding chunk #{chunk_count} from file {file_id}")
                yield ascii_data
                
            except Exception as e:
                logging.warning(f"Error processing LIBRISPEECH file {file_id}: {e}")
                continue
        
        logging.debug(f"LIBRISPEECH data generator completed: {chunk_count} chunks from {files_processed} files")
    
    return librispeech_data_generator()


# Import all the core functions from zero_shot.py
# We'll import them dynamically to avoid circular imports
def import_zero_shot_functions():
    """Import core functions from zero_shot.py"""
    import importlib.util
    import sys
    
    # Load zero_shot.py as a module
    spec = importlib.util.spec_from_file_location("zero_shot", "/home/pnlong/lnac/lmic/zero_shot.py")
    zero_shot_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(zero_shot_module)
    
    # Import the functions we need
    global load_llama_model, load_framework_model, load_model_parameters
    global create_model_predict_fn, create_audio_processor_fn
    global evaluate_language_model, evaluate_baseline_compressors
    
    load_llama_model = zero_shot_module.load_llama_model
    load_framework_model = zero_shot_module.load_framework_model
    load_model_parameters = zero_shot_module.load_model_parameters
    create_model_predict_fn = zero_shot_module.create_model_predict_fn
    create_audio_processor_fn = zero_shot_module.create_audio_processor_fn
    evaluate_language_model = zero_shot_module.evaluate_language_model
    evaluate_baseline_compressors = zero_shot_module.evaluate_baseline_compressors


def evaluate_librispeech_baseline_compressors(
    compressors: List[str],
    args: argparse.Namespace,
    bit_depth: int
) -> Dict[str, Dict[str, Any]]:
    """Evaluate baseline compressors on LIBRISPEECH raw audio data."""
    results = {}
    
    for compressor_name in compressors:
        logging.info(f"Evaluating baseline compressor: {compressor_name}")
        
        try:
            # Create LIBRISPEECH raw audio generator (like zero_shot.py)
            raw_data_generator = create_librispeech_raw_audio_generator(args, bit_depth)
            
            # Choose appropriate compression function
            if compressor_name == 'flac':
                # FLAC works best on continuous audio streams
                compress_fn = create_flac_compress_fn(bit_depth)
                compression_ratio, compression_time = evaluate_compressor_unchunked(
                    compress_fn=compress_fn,
                    get_data_generator_fn=lambda: raw_data_generator,
                    num_chunks=args.num_chunks,
                )
            else:
                # GZIP/LZMA work fine on chunks
                compress_fn = compressor.COMPRESS_FN_DICT[compressor_name]
                compression_ratio, compression_time = evaluate_compressor_chunked(
                    compress_fn=compress_fn,
                    get_data_generator_fn=lambda: raw_data_generator,
                    num_chunks=args.num_chunks,
                    count_header_only_once=True,
                    mask_fn=None,  # No masking for raw data
                    use_tqdm=True,
                )
            
            # Safety check for division by zero
            if compression_ratio == float('inf') or compression_ratio != compression_ratio:  # NaN check
                logging.error(f"Invalid compression ratio for {compressor_name}: {compression_ratio}")
                results[compressor_name] = {
                    "error": f"Invalid compression ratio: {compression_ratio}",
                    "compressor_type": "baseline",
                    "data_type": "raw_audio"
                }
            else:
                results[compressor_name] = {
                    "compression_ratio": compression_ratio,
                    "compression_time": compression_time,
                    "compressor_type": "baseline",
                    "data_type": "raw_audio"
                }
            
        except Exception as e:
            logging.warning(f"Failed to evaluate {compressor_name}: {str(e)}")
            results[compressor_name] = {"error": str(e)}
    
    return results


def create_flac_compress_fn(bit_depth: int) -> Callable[[bytes], bytes]:
    """Create FLAC compression function with proper bit depth handling."""
    def flac_compress(data: bytes) -> bytes:
        # Import FLAC compressor
        from language_modeling_is_compression.compressors import flac
        return flac.compress(data, bit_depth=bit_depth)
    
    return flac_compress


def format_librispeech_results(results: Dict[str, Any], args: argparse.Namespace) -> str:
    """Format LIBRISPEECH evaluation results for display."""
    output = []
    output.append("Zero-Shot Language Model Evaluation Results (LIBRISPEECH)")
    output.append("=" * 60)
    output.append("")
    
    # Configuration section
    output.append("Configuration:")
    output.append(f"  LIBRISPEECH Split: {args.librispeech_split}")
    output.append(f"  Model: {args.model_path}")
    output.append(f"  Bit Depth: {results.get('language_model', {}).get('bit_depth', 'Unknown')} (detected from audio)")
    output.append(f"  Chunk Size: {args.chunk_size} samples (mono audio)")
    output.append(f"  Number of Chunks: {args.num_chunks}")
    output.append(f"  Chunks Per File: {args.chunks_per_file}")
    if args.max_files:
        output.append(f"  Max Files: {args.max_files}")
    output.append(f"  Random Seed: {args.random_seed}")
    output.append("")
    
    # Results section
    output.append("Results:")
    
    # Language model results
    if "language_model" in results:
        lm_result = results["language_model"]
        if "error" not in lm_result:
            ratio = lm_result["compression_ratio"] * 100
            time_taken = lm_result["compression_time"]
            model_type = lm_result.get("model_type", "framework")
            data_type = lm_result.get("data_type", "processed_ascii")
            output.append(f"  Language Model ({model_type}): {ratio:.1f}% compression ratio ({time_taken:.1f}s) [Data: {data_type}]")
        else:
            output.append(f"  Language Model: ERROR - {lm_result['error']}")
    
    # Baseline results
    for compressor_name, result in results.items():
        if compressor_name != "language_model":
            if "error" not in result:
                ratio = result["compression_ratio"] * 100
                time_taken = result["compression_time"]
                data_type = result.get("data_type", "raw_audio")
                output.append(f"  {compressor_name.upper()} Baseline: {ratio:.1f}% compression ratio ({time_taken:.1f}s) [Data: {data_type}]")
            else:
                output.append(f"  {compressor_name.upper()} Baseline: ERROR - {result['error']}")
    
    output.append("")
    
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


def save_librispeech_results(results: Dict[str, Any], args: argparse.Namespace, output_file: str, bit_depth: int) -> None:
    """Save detailed LIBRISPEECH results to file."""
    # Prepare results for JSON serialization
    json_results = {
        "configuration": {
            "librispeech_split": args.librispeech_split,
            "model_path": args.model_path,
            "bit_depth": bit_depth,
            "chunk_size": args.chunk_size,
            "num_chunks": args.num_chunks,
            "chunks_per_file": args.chunks_per_file,
            "embedding_dim": args.embedding_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "widening_factor": args.widening_factor,
            "max_files": args.max_files,
            "random_seed": args.random_seed,
        },
        "results": results,
        "summary": {}
    }
    
    # Add summary information
    if "language_model" in results and "error" not in results["language_model"]:
        lm_ratio = results["language_model"]["compression_ratio"]
        lm_model_type = results["language_model"].get("model_type", "framework")
        json_results["summary"]["best_compressor"] = "language_model"
        json_results["summary"]["best_ratio"] = lm_ratio
        json_results["summary"]["model_type"] = lm_model_type
        
        # Calculate improvements
        improvements = {}
        baseline_results = {k: v for k, v in results.items() if k != "language_model" and "error" not in v}
        for baseline_name, baseline_result in baseline_results.items():
            improvement = (baseline_result["compression_ratio"] - lm_ratio) / baseline_result["compression_ratio"]
            improvements[f"vs_{baseline_name}"] = improvement
        
        json_results["summary"]["improvements"] = improvements
        json_results["summary"]["total_evaluation_time"] = sum(
            r.get("total_time", 0) for r in results.values() if "error" not in r
        )
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logging.info(f"LIBRISPEECH results saved to {output_file}")


def run_comprehensive_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Run complete evaluation with LIBRISPEECH dataset."""
    results = {}
    
    # Import zero_shot functions
    import_zero_shot_functions()
    
    # Detect bit depth from LIBRISPEECH files
    audio_files = get_librispeech_files(args.librispeech_split, args.max_files, args.random_seed)
    bit_depth = detect_bit_depth_from_librispeech(audio_files)
    logging.info(f"Detected bit depth: {bit_depth}")
    
    # Validate bit depth is 8-bit as expected
    if bit_depth != 8:
        logging.warning(f"Expected 8-bit audio but detected {bit_depth}-bit. Proceeding with detected bit depth.")
    
    # Create single raw audio generator for all compressors (like zero_shot.py)
    logging.info("Creating unified LIBRISPEECH raw audio generator...")
    raw_data_generator = create_librispeech_raw_audio_generator(args, bit_depth)
    
    # Evaluate language model (uses raw audio with on-the-fly processing)
    try:
        lm_results = evaluate_language_model(args.model_path, raw_data_generator, args, bit_depth)
        results["language_model"] = lm_results
    except Exception as e:
        logging.error(f"Failed to evaluate language model: {str(e)}")
        results["language_model"] = {"error": str(e)}
    
    # Evaluate baseline compressors (uses same raw audio data)
    if args.baseline_compressors:
        baseline_results = evaluate_librispeech_baseline_compressors(args.baseline_compressors, args, bit_depth)
        results.update(baseline_results)
    
    return results


def main() -> None:
    """Main function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(args.verbose)
        
        # Validate arguments
        validate_arguments(args)
        
        # Log configuration
        logging.info("Starting zero-shot language model evaluation on LIBRISPEECH dataset")
        logging.info(f"Configuration: {vars(args)}")
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        if torch is not None:
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
        
        # Run evaluation
        results = run_comprehensive_evaluation(args)
        
        # Format and display results
        formatted_results = format_librispeech_results(results, args)
        print(formatted_results)
        
        # Save results if requested
        if args.output_file:
            bit_depth = results.get('language_model', {}).get('bit_depth', 8)
            save_librispeech_results(results, args, args.output_file, bit_depth)
        
        logging.info("LIBRISPEECH evaluation completed successfully!")
        
    except Exception as e:
        logging.error(f"LIBRISPEECH evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
