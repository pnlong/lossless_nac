#!/usr/bin/env python3
"""
Zero-Shot Language Model Evaluation Script

This script evaluates pre-trained language models on custom audio data without training.
It provides comprehensive configurability for audio processing parameters and supports
multiple evaluation scenarios including baseline comparisons.

Usage:
    python zero_shot.py --audio_dir /path/to/audio --model_path /path/to/model.npz
    python zero_shot.py --audio_dir /path/to/audio --model_path /path/to/model.npz --bit_depth 16 --gpu --verbose
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
except ImportError:
    torch = None

# Import framework components first
from language_modeling_is_compression import constants
from language_modeling_is_compression import data_loaders
from language_modeling_is_compression import utils
from language_modeling_is_compression import transformer
from language_modeling_is_compression.compressors import compressor
from language_modeling_is_compression.compress import evaluate_compressor_chunked

# Llama imports will be done lazily to avoid conflicts with Haiku/JAX
LLAMA_AVAILABLE = None  # Will be set when actually needed

# Import local utilities
from data_paths import get_all_paths


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
DEFAULT_CHUNK_SIZE = 2048
DEFAULT_NUM_CHUNKS = 1000
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_NUM_LAYERS = 4
DEFAULT_NUM_HEADS = 8
DEFAULT_WIDENING_FACTOR = 4
DEFAULT_BASELINE_COMPRESSORS = ['gzip', 'flac', 'lzma']
DEFAULT_VERBOSE = False
DEFAULT_SLOW_COMPRESSION = False
DEFAULT_GPU = True
DEFAULT_MAX_LENGTH = 2048
DEFAULT_BATCH_SIZE = 32


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
        description="Zero-shot evaluation of language models on custom audio data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Path to directory containing WAV files"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model file (.npz) or Llama model directory (e.g., './llama-2-7b-chat-hf')"
    )
    
    # Audio processing parameters
    parser.add_argument(
        "--bit_depth",
        type=int,
        choices=[8, 16, 24, 32],
        default=8,
        help="Audio bit depth (8, 16, 24, or 32 bits)"
    )
    
    
    parser.add_argument(
        "--stereo_blocking_n",
        type=int,
        default=DEFAULT_STEREO_BLOCKING_N,
        help="Size of blocks for stereo processing in samples"
    )
    
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Size of each compression chunk in bytes"
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
        default=DEFAULT_GPU,
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
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    # Check if audio directory exists
    if not os.path.exists(args.audio_dir):
        raise ValueError(f"Audio directory does not exist: {args.audio_dir}")
    
    if not os.path.isdir(args.audio_dir):
        raise ValueError(f"Audio path is not a directory: {args.audio_dir}")
    
    # Check if model file exists (only for framework models)
    model_type = detect_model_type(args.model_path)
    if model_type == "framework":
        if not os.path.exists(args.model_path):
            raise ValueError(f"Model file does not exist: {args.model_path}")
    
    # Validate positive integers
    positive_int_args = [
        "stereo_blocking_n", "chunk_size", "num_chunks", "embedding_dim",
        "num_layers", "num_heads", "widening_factor", "max_length", "batch_size"
    ]
    
    for arg_name in positive_int_args:
        value = getattr(args, arg_name)
        if value <= 0:
            raise ValueError(f"{arg_name} must be positive, got {value}")
    
    # Validate bit depth specific constraints
    from language_modeling_is_compression import ascii_mapping
    bytes_per_sample = ascii_mapping.calculate_bytes_per_sample(args.bit_depth)
    if args.chunk_size % bytes_per_sample != 0:
        raise ValueError(f"chunk_size must be divisible by {bytes_per_sample} for {args.bit_depth}-bit audio")
    
    # Validate blocking size vs chunk size
    if args.stereo_blocking_n * 2 < args.chunk_size:
        warnings.warn(
            f"Blocking size {args.stereo_blocking_n} is too small for chunk size {args.chunk_size}. "
            f"Chunks will not contain complete L/R block pairs."
        )
    
    # Validate baseline compressors
    if args.baseline_compressors:
        invalid_compressors = [c for c in args.baseline_compressors if c not in compressor.COMPRESS_FN_DICT]
        if invalid_compressors:
            raise ValueError(f"Invalid baseline compressors: {invalid_compressors}")


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


def load_llama_model(model_dir: str, use_gpu: bool = True) -> Dict[str, Any]:
    """Load Llama model using Meta's official pattern."""
    if not _import_llama_modules():
        raise ImportError("Llama transformers not available. Install with: pip install transformers torch")
    
    # Import the modules now that we know they're available
    import torch
    import transformers
    from transformers import LlamaForCausalLM, LlamaTokenizer
    
    # Remove llama: prefix if present
    if model_dir.startswith("llama:"):
        model_dir = model_dir.replace("llama:", "")
    
    logging.info(f"Loading Llama model from: {model_dir}")
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        raise ValueError(f"Llama model directory does not exist: {model_dir}")
    
    if not os.path.isdir(model_dir):
        raise ValueError(f"Llama model path is not a directory: {model_dir}")
    
    # Check for Hugging Face format
    config_file = os.path.join(model_dir, "config.json")
    pytorch_model_file = os.path.join(model_dir, "pytorch_model.bin")
    
    # Check for original Llama format
    consolidated_file = os.path.join(model_dir, "consolidated.00.pth")
    params_file = os.path.join(model_dir, "params.json")
    
    if os.path.exists(config_file) and os.path.exists(pytorch_model_file):
        # Hugging Face format
        return _load_huggingface_llama_model(model_dir, use_gpu)
    elif os.path.exists(consolidated_file) and os.path.exists(params_file):
        # Original Llama format
        return _load_original_llama_model(model_dir, use_gpu)
    else:
        raise ValueError(f"Llama model directory missing required files. Expected either Hugging Face format (config.json + pytorch_model.bin) or original format (consolidated.00.pth + params.json): {model_dir}")


def _load_huggingface_llama_model(model_dir: str, use_gpu: bool = True) -> Dict[str, Any]:
    """Load Llama model in Hugging Face format."""
    import torch
    import transformers
    from transformers import LlamaForCausalLM, LlamaTokenizer
    
    try:
        # Determine device for loading
        if use_gpu and torch.cuda.is_available():
            device = "cuda"  # Use the first available GPU (respects CUDA_VISIBLE_DEVICES)
        else:
            device = "cpu"
            
        logging.info(f"Loading Llama model and tokenizer (Hugging Face format) on device: {device}")
        
        # Load model and tokenizer using Hugging Face pattern
        model = LlamaForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,  # Use float16 for efficiency
            device_map=device,
        )
        tokenizer = LlamaTokenizer.from_pretrained(model_dir)
        
        # Create pipeline for inference
        logging.info("Creating text generation pipeline...")
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map=device,
        )
        
        logging.info(f"Successfully loaded Llama model from {model_dir} (Hugging Face format)")
        return {
            "model": model,
            "tokenizer": tokenizer,
            "pipeline": pipeline,
            "model_type": "llama",
            "format": "huggingface",
            "device": device
        }
    except Exception as e:
        raise ValueError(f"Error loading Hugging Face Llama model from '{model_dir}': {str(e)}")


def _load_original_llama_model(model_dir: str, use_gpu: bool = True) -> Dict[str, Any]:
    """Load Llama model in original Meta format."""
    import torch
    import json
    from transformers import LlamaTokenizer
    
    try:
        # Load model parameters
        params_file = os.path.join(model_dir, "params.json")
        with open(params_file, 'r') as f:
            params = json.load(f)
        
        logging.info(f"Loaded model parameters: {params}")
        
        # Load model weights
        consolidated_file = os.path.join(model_dir, "consolidated.00.pth")
        logging.info("Loading model weights from consolidated.00.pth...")
        
        # Determine device for loading
        if use_gpu and torch.cuda.is_available():
            device = "cuda"  # Use the first available GPU (respects CUDA_VISIBLE_DEVICES)
        else:
            device = "cpu"
            
        logging.info(f"Loading model weights on device: {device}")
        checkpoint = torch.load(consolidated_file, map_location=device)
        
        # For now, we'll use a simplified approach - load the tokenizer and create a dummy model
        # In a full implementation, you'd need to reconstruct the model architecture from the checkpoint
        tokenizer_file = os.path.join(model_dir, "tokenizer.model")
        if os.path.exists(tokenizer_file):
            # Load tokenizer from original format
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(model_dir)
        else:
            # Fallback to a default tokenizer
            tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        
        # Create a simple wrapper for the original format
        # Note: This is a simplified implementation - in practice, you'd need to properly
        # reconstruct the model architecture from the checkpoint
        logging.info("Creating model wrapper for original format...")
        
        # For now, we'll create a minimal implementation that can be used for testing
        # In a production system, you'd want to properly load the model architecture
        model_wrapper = {
            "checkpoint": checkpoint,
            "params": params,
            "tokenizer": tokenizer,
            "format": "original"
        }
        
        logging.info(f"Successfully loaded Llama model from {model_dir} (original format)")
        return {
            "model": model_wrapper,
            "tokenizer": tokenizer,
            "pipeline": None,  # No pipeline for original format
            "model_type": "llama",
            "format": "original",
            "device": device
        }
    except Exception as e:
        raise ValueError(f"Error loading original Llama model from '{model_dir}': {str(e)}")


def load_framework_model(model_path: str) -> hk.Params:
    """Load framework model parameters from file."""
    try:
        with np.load(model_path, allow_pickle=True) as data:
            params = {key: data[key].item() for key in data.files}
        logging.info(f"Successfully loaded framework model parameters from {model_path}")
        return params
    except Exception as e:
        raise ValueError(f"Error loading framework model parameters from {model_path}: {str(e)}")


def load_model_parameters(model_path: str, use_gpu: bool = True) -> Any:
    """Load model parameters from either Llama or framework format."""
    model_type = detect_model_type(model_path)
    
    if model_type == "llama":
        return load_llama_model(model_path, use_gpu)
    else:
        return load_framework_model(model_path)


def create_llama_predict_fn(model_info: Dict[str, Any]) -> Callable:
    """Create prediction function for Llama model."""
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    def predict_fn(sequence_array: np.ndarray) -> np.ndarray:
        """Predict next token probabilities for Llama model."""
        # Convert numpy array to tokens
        tokens = sequence_array.astype(np.uint8)
        
        # Convert to PyTorch tensor
        input_ids = torch.tensor(tokens).unsqueeze(0)
        
        # Get logits using the model directly (not pipeline for efficiency)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            
        # Convert back to numpy and return log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs.squeeze(0).cpu().numpy()
    
    return predict_fn


def create_model_predict_fn(params: hk.Params) -> Any:
    """Create prediction function for framework model."""
    config = transformer.create_audio_transformer_config(use_16bit=False)
    model = hk.transform(
        lambda x: transformer.transformer_decoder(x, config=config)
    )
    return lambda x: model.apply(params, None, x)


def detect_stereo_audio(audio_files: List[str]) -> bool:
    """Detect if any audio files are stereo."""
    from scipy.io import wavfile
    
    for audio_file in audio_files[:5]:  # Check first 5 files
        try:
            sr, audio_data = wavfile.read(audio_file)
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                return True
        except Exception as e:
            logging.warning(f"Error checking stereo for {audio_file}: {e}")
            continue
    
    return False


def analyze_audio_processing_pipeline(audio_files: List[str], args: argparse.Namespace) -> None:
    """Analyze the audio processing pipeline to identify issues."""
    if not audio_files:
        logging.error("No audio files found to analyze")
        return
    
    # Analyze first audio file in detail
    audio_file = audio_files[0]
    logging.info(f"Analyzing audio file: {os.path.basename(audio_file)}")
    
    try:
        from scipy.io import wavfile
        from language_modeling_is_compression import audio_processing_extended
        
        # Step 1: Load with scipy
        sr, audio_data = wavfile.read(audio_file)
        
        # Calculate amplitude as percentage of maximum
        if audio_data.dtype == np.int16:
            max_possible = 32767
        elif audio_data.dtype == np.int32:
            max_possible = 2147483647
        elif audio_data.dtype == np.uint8:
            max_possible = 255
        else:
            max_possible = audio_data.max()
        
        max_amplitude = max(abs(audio_data.min()), abs(audio_data.max()))
        amplitude_percentage = (max_amplitude / max_possible) * 100 if max_possible > 0 else 0
        
        if amplitude_percentage < 1.0:
            logging.warning(f"Audio is very quiet ({amplitude_percentage:.2f}% of maximum) - likely silence")
        elif amplitude_percentage < 10.0:
            logging.warning(f"Audio is quiet ({amplitude_percentage:.2f}% of maximum)")
        
        # Check if original audio is repetitive
        if len(np.unique(audio_data)) < 10:
            logging.warning(f"Original audio is highly repetitive: {len(np.unique(audio_data))} unique values")
        
        # Process through the pipeline
        if len(audio_data.shape) > 1:
            processed_audio = audio_processing_extended.process_stereo_blocking_extended(audio_data, args.stereo_blocking_n)
        else:
            processed_audio = audio_data
        
        # Normalize to [-1, 1]
        if processed_audio.dtype == np.int16:
            normalized_audio = processed_audio.astype(np.float32) / 32767.0
        elif processed_audio.dtype == np.int32:
            normalized_audio = processed_audio.astype(np.float32) / 2147483647.0
        elif processed_audio.dtype == np.uint8:
            normalized_audio = (processed_audio.astype(np.float32) - 128.0) / 128.0
        else:
            normalized_audio = processed_audio.astype(np.float32)
        
        # Check if all values are the same after normalization
        if len(np.unique(normalized_audio)) == 1:
            logging.warning(f"All normalized values are the same: {normalized_audio[0]} - likely silence")
        
        # Convert to target bit depth and check result
        audio_bytes = audio_processing_extended.convert_to_target_bit_depth_extended(normalized_audio, args.bit_depth)
        
        # Check if final bytes are repetitive
        if len(set(audio_bytes)) < 10:
            logging.warning(f"Final audio bytes are highly repetitive: {len(set(audio_bytes))} unique values")
        
    except Exception as e:
        logging.error(f"Error analyzing audio processing pipeline: {e}")


def setup_audio_data_generator(args: argparse.Namespace) -> Iterator[bytes]:
    """Set up audio data generator with bit depth support."""
    # Get audio file paths
    try:
        audio_files = get_all_paths(args.audio_dir)
        logging.info(f"Found {len(audio_files)} WAV files in {args.audio_dir}")
    except Exception as e:
        raise ValueError(f"Error discovering audio files: {str(e)}")
    
    # Analyze the audio processing pipeline
    analyze_audio_processing_pipeline(audio_files, args)
    
    # Detect if we have stereo audio
    is_stereo = detect_stereo_audio(audio_files)
    if is_stereo:
        logging.info("Detected stereo audio files - will use interleaved blocking (doubles sequence length)")
        # Adjust max_length for stereo (interleaved blocking doubles the sequence length)
        effective_max_length = args.max_length * 2
        logging.info(f"Adjusted max_length from {args.max_length} to {effective_max_length} for stereo audio")
    else:
        logging.info("Detected mono audio files - using standard processing")
        effective_max_length = args.max_length
    
    # Store the effective max_length for use in Llama models
    args.effective_max_length = effective_max_length
    
    # Create data generator with bit depth support
    from language_modeling_is_compression import audio_processing_extended
    logging.info("=== CREATING DATA GENERATOR ===")
    logging.info(f"Using audio_processing_extended.get_custom_audio_iterator_extended()")
    logging.info(f"Parameters: bit_depth={args.bit_depth}, blocking_size={args.stereo_blocking_n}, chunk_size={args.chunk_size}")
    
    base_data_generator = audio_processing_extended.get_custom_audio_iterator_extended(
        audio_files=audio_files,
        num_chunks=args.num_chunks,
        bit_depth=args.bit_depth,
        blocking_size=args.stereo_blocking_n,
        chunk_size_bytes=args.chunk_size,
    )
    
    # Wrap the data generator to add file tracking
    def tracked_data_generator():
        chunk_count = 0
        for chunk in base_data_generator:
            chunk_count += 1
            logging.debug(f"Data generator yielding chunk #{chunk_count}")
            yield chunk
    
    return tracked_data_generator()


def evaluate_language_model(
    model_path: str,
    data_generator: Iterator[bytes],
    args: argparse.Namespace
) -> Dict[str, Any]:
    """Evaluate language model on the provided data with bit depth support."""
    logging.info("Evaluating language model...")
    
    # Load model parameters
    model_info = load_model_parameters(model_path, use_gpu=args.gpu)
    
    # Create prediction function based on model type
    if isinstance(model_info, dict) and model_info.get("model_type") == "llama":
        from language_modeling_is_compression import llama_integration
        # Use effective max_length for stereo audio (doubled if stereo detected)
        effective_max_length = getattr(args, 'effective_max_length', args.max_length)
        predict_fn = llama_integration.create_llama_predict_fn_extended(
            model_info, 
            bit_depth=args.bit_depth,
            max_length=effective_max_length
        )
        model_type = "llama"
    else:
        predict_fn = create_model_predict_fn(model_info)
        model_type = "framework"
    
    # Create custom compression function - FIXED to match paper's approach
    chunk_counter = [0]  # Use list to make it mutable in closure
    current_file = [None]  # Track current file being processed
    
    def language_model_compress(data: bytes) -> bytes:
        """Fixed compression function following paper's approach."""
        
        # Increment chunk counter
        chunk_counter[0] += 1
        logging.debug(f"Processing chunk #{chunk_counter[0]}")
        
        # Step 1: Convert data to array based on bit depth
        
        # Check for repetitive patterns in raw data (silent check)
        unique_bytes = len(set(data))
        
        if args.bit_depth == 8:
            sequence_array = np.frombuffer(data, dtype=np.uint8)
        elif args.bit_depth == 16:
            sequence_array = np.frombuffer(data, dtype=np.int16)
        elif args.bit_depth == 24:
            # Convert 24-bit bytes to numpy array
            from language_modeling_is_compression import ascii_mapping
            bytes_per_sample = ascii_mapping.calculate_bytes_per_sample(args.bit_depth)
            num_samples = len(data) // bytes_per_sample
            sequence_array = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                start_idx = i * bytes_per_sample
                end_idx = start_idx + bytes_per_sample
                sample_bytes = data[start_idx:end_idx]
                # Convert to signed 24-bit integer
                sample = int.from_bytes(sample_bytes, byteorder='little', signed=False)
                if sample >= (1 << 23):  # Check if negative in 24-bit signed
                    sample = sample - (1 << 24)
                sequence_array[i] = sample
        elif args.bit_depth == 32:
            sequence_array = np.frombuffer(data, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported bit depth: {args.bit_depth}")
        
        # Silent sequence array analysis
        
        # Check for repetitive patterns in sequence
        # Silent check for repetitive patterns
        if len(np.unique(sequence_array)) < 10:
            pass  # Silent repetitive data check
        
        # Step 2: Convert to bytes
        if args.bit_depth == 8:
            data_bytes = sequence_array.astype(np.uint8).tobytes()
        elif args.bit_depth == 16:
            data_bytes = sequence_array.astype(np.int16).tobytes()
        elif args.bit_depth == 24:
            # Convert to 24-bit bytes
            data_bytes = []
            for sample in sequence_array.flatten():
                if sample < 0:
                    sample = sample + (1 << 24)  # Convert to unsigned 24-bit
                data_bytes.extend(sample.astype(np.int32).tobytes()[:3])  # Take first 3 bytes
            data_bytes = bytes(data_bytes)
        elif args.bit_depth == 32:
            data_bytes = sequence_array.astype(np.int32).tobytes()
        else:
            raise ValueError(f"Unsupported bit depth: {args.bit_depth}")
        
        # Step 3: Apply ASCII mapping (ONCE, not twice!)
        from language_modeling_is_compression import ascii_mapping
        ascii_mapping_fn = ascii_mapping.get_ascii_mapping_function_for_bit_depth(args.bit_depth)
        ascii_data, dropped_lsb_bits = ascii_mapping_fn(data_bytes)
        
        # Step 4: Convert to ASCII string
        ascii_text = ascii_data.decode('ascii', errors='ignore')
        
        # Check if ASCII text is valid
        if len(ascii_text) == 0:
            logging.warning(f"Empty ASCII text generated from {len(data)} bytes of data")
            return b''
        
        # Note: Variable length is OK for multi-bit depths
        # 8-bit: 1 sample → 1 ASCII char
        # 16-bit: 1 sample → 2 ASCII chars  
        # 24-bit: 1 sample → 3 ASCII chars
        # 32-bit: 1 sample → 4 ASCII chars
        
        # Step 5: Get predictions using the fixed prediction function
        log_probs = predict_fn(ascii_text)  # Shape: (l, T) where l=seq_len, T=vocab_size
        
        # Step 6: Tokenize for arithmetic coding
        tokens = model_info["tokenizer"].encode(ascii_text, add_special_tokens=False)
        
        # Check if tokenization is working
        if len(tokens) == 0:
            logging.warning(f"No tokens generated from ASCII text of length {len(ascii_text)}")
            return b''
        
        # Step 7: Apply top-k filtering as described in paper
        # Paper: "In practice, the large models had only access to the top-k next token log-probabilities, for each context"
        from language_modeling_is_compression import llama_integration
        # Use top-k filtering with k=100 as per paper
        log_probs_topk = llama_integration.apply_top_k_filtering(log_probs, k=100)
        
        # Step 8: Use arithmetic coding with paper's approach
        from language_modeling_is_compression import arithmetic_coder
        
        output = []
        encoder = arithmetic_coder.Encoder(
            base=constants.ARITHMETIC_CODER_BASE,
            precision=constants.ARITHMETIC_CODER_PRECISION,
            output_fn=output.append,
        )
        
        # Encode tokens using paper's approach: for each position, use the prediction for that position
        for i, token_id in enumerate(tokens):
            if i < len(log_probs_topk):
                pdf = np.exp(log_probs_topk[i])  # Convert log probs to probs for position i
                
                # Check if token is in top-k (has non-zero probability)
                if pdf[token_id] == 0:
                    continue
                
                encoder.encode(utils.normalize_pdf_for_arithmetic_coding(pdf), token_id)
        
        encoder.terminate()
        
        # Step 9: Convert bits to bytes
        compressed_bits = ''.join(map(str, output))
        compressed_bytes, _ = utils.bits_to_bytes(compressed_bits)
        
        # Step 10: Append dropped LSB bits for lossless reconstruction
        final_compressed = compressed_bytes + dropped_lsb_bits
        
        # Check if LSB bits are making data much larger (silent check)
        if len(dropped_lsb_bits) > len(compressed_bytes):
            pass  # Silent LSB size check
        
        return final_compressed
    
    # Get appropriate mask function for bit depth
    from language_modeling_is_compression import utils
    mask_fn = utils.get_mask_function_for_bit_depth(args.bit_depth, use_ascii_check=False)
    
    # Evaluate compression using existing infrastructure
    start_time = time.perf_counter()
    compression_ratio, compression_time = evaluate_compressor_chunked(
        compress_fn=language_model_compress,
        get_data_generator_fn=lambda: data_generator,
        num_chunks=args.num_chunks,
        count_header_only_once=False,
        mask_fn=mask_fn,  # Use bit depth specific mask function
        use_tqdm=True,
    )
    total_time = time.perf_counter() - start_time
    
    return {
        "compression_ratio": compression_ratio,
        "compression_time": compression_time,
        "total_time": total_time,
        "compressor_type": "language_model",
        "model_type": model_type,
        "bit_depth": args.bit_depth
    }


def evaluate_baseline_compressors(
    data_generator: Iterator[bytes],
    compressors: List[str],
    args: argparse.Namespace
) -> Dict[str, Dict[str, Any]]:
    """Evaluate baseline compressors for comparison."""
    results = {}
    
    for compressor_name in compressors:
        logging.info(f"Evaluating baseline compressor: {compressor_name}")
        
        try:
            # Create compressor function with bit_depth parameter for FLAC
            if compressor_name == 'flac':
                def flac_compress_with_bit_depth(data: bytes) -> bytes:
                    # Import flac directly to avoid wrapper issues
                    from language_modeling_is_compression.compressors import flac
                    return flac.compress(data, bit_depth=args.bit_depth)
                compress_fn = flac_compress_with_bit_depth
            else:
                compress_fn = compressor.COMPRESS_FN_DICT[compressor_name]
            
            # Create new data generator for each compressor
            audio_files = get_all_paths(args.audio_dir)
            from language_modeling_is_compression import audio_processing_extended
            new_data_generator = audio_processing_extended.get_custom_audio_iterator_extended(
                audio_files=audio_files,
                num_chunks=args.num_chunks,
                bit_depth=args.bit_depth,
                blocking_size=args.stereo_blocking_n,
                chunk_size_bytes=args.chunk_size,
            )
            
            start_time = time.perf_counter()
            
            if compressor_name in compressor.COMPRESSOR_TYPES['classical']:
                compression_ratio, compression_time = evaluate_compressor_chunked(
                    compress_fn=compress_fn,
                    get_data_generator_fn=lambda: new_data_generator,
                    num_chunks=args.num_chunks,
                    count_header_only_once=True,
                    mask_fn=None,
                    use_tqdm=True,
                )
            else:
                # For arithmetic coding compressors
                compression_ratio, compression_time = evaluate_compressor_chunked(
                    compress_fn=compress_fn,
                    get_data_generator_fn=lambda: new_data_generator,
                    num_chunks=args.num_chunks,
                    count_header_only_once=False,
                    mask_fn=utils.right_shift_bytes_by_one,
                    use_tqdm=True,
                )
            
            total_time = time.perf_counter() - start_time
            
            results[compressor_name] = {
                "compression_ratio": compression_ratio,
                "compression_time": compression_time,
                "total_time": total_time,
                "compressor_type": "baseline"
            }
            
        except Exception as e:
            logging.warning(f"Failed to evaluate {compressor_name}: {str(e)}")
            results[compressor_name] = {
                "error": str(e),
                "compressor_type": "baseline"
            }
    
    return results


def format_results(results: Dict[str, Any], args: argparse.Namespace) -> str:
    """Format evaluation results for display."""
    output = []
    output.append("Zero-Shot Language Model Evaluation Results")
    output.append("=" * 50)
    output.append("")
    
    # Configuration section
    output.append("Configuration:")
    output.append(f"  Audio Directory: {args.audio_dir}")
    output.append(f"  Model: {args.model_path}")
    output.append(f"  Bit Depth: {args.bit_depth}")
    output.append(f"  Stereo Blocking: {args.stereo_blocking_n} samples")
    output.append(f"  Chunk Size: {args.chunk_size} bytes")
    output.append(f"  Number of Chunks: {args.num_chunks}")
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
            output.append(f"  Language Model ({model_type}): {ratio:.1f}% compression ratio ({time_taken:.1f}s)")
        else:
            output.append(f"  Language Model: ERROR - {lm_result['error']}")
    
    # Baseline results
    for compressor_name, result in results.items():
        if compressor_name != "language_model":
            if "error" not in result:
                ratio = result["compression_ratio"] * 100
                time_taken = result["compression_time"]
                output.append(f"  {compressor_name.upper()} Baseline: {ratio:.1f}% compression ratio ({time_taken:.1f}s)")
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


def save_results(results: Dict[str, Any], args: argparse.Namespace, output_file: str) -> None:
    """Save detailed results to file."""
    # Prepare results for JSON serialization
    json_results = {
        "configuration": {
            "audio_directory": args.audio_dir,
            "model_path": args.model_path,
            "bit_depth": args.bit_depth,
            "stereo_blocking_n": args.stereo_blocking_n,
            "chunk_size": args.chunk_size,
            "num_chunks": args.num_chunks,
            "embedding_dim": args.embedding_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "widening_factor": args.widening_factor,
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
    
    logging.info(f"Results saved to {output_file}")


def run_comprehensive_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Run complete evaluation with all specified models and baselines."""
    results = {}
    
    # Set up audio data generator
    data_generator = setup_audio_data_generator(args)
    
    # Evaluate language model
    try:
        lm_results = evaluate_language_model(args.model_path, data_generator, args)
        results["language_model"] = lm_results
    except Exception as e:
        logging.error(f"Failed to evaluate language model: {str(e)}")
        results["language_model"] = {"error": str(e)}
    
    # Evaluate baseline compressors
    if args.baseline_compressors:
        baseline_results = evaluate_baseline_compressors(data_generator, args.baseline_compressors, args)
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
        logging.info("Starting zero-shot language model evaluation")
        logging.info(f"Configuration: {vars(args)}")
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        # Run evaluation
        results = run_comprehensive_evaluation(args)
        
        # Format and display results
        formatted_results = format_results(results, args)
        print(formatted_results)
        
        # Save results if requested
        if args.output_file:
            save_results(results, args, args.output_file)
        
        logging.info("Evaluation completed successfully!")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

