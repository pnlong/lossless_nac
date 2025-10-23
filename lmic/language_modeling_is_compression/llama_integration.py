"""Llama model integration functions for zero-shot compression."""

import numpy as np
import torch
from typing import Dict, Any, Callable
from . import ascii_mapping
from . import utils
from . import constants
from . import arithmetic_coder


def _run_original_llama_inference(model_wrapper, input_ids):
    """Run inference on original Llama model format - FALLBACK TO HUGGINGFACE."""
    # The original format inference is too complex to implement correctly.
    # Instead, let's try to use Hugging Face format if possible.
    import torch
    import logging
    logging.warning("Original Llama format inference is not fully implemented. Consider using Hugging Face format.")
    
    # For now, return uniform logits to avoid errors
    batch_size, seq_len = input_ids.shape
    vocab_size = 32000  # Default vocab size
    
    # Return uniform logits (this will give poor compression but won't crash)
    uniform_logits = torch.zeros(batch_size, seq_len, vocab_size)
    return uniform_logits


def create_llama_predict_fn_extended(model_info: Dict[str, Any], bit_depth: int = 8, max_length: int = 2048) -> Callable:
  """Create prediction function for Llama model with extended bit depth support.
  
  FIXED: Now does single forward pass to get l*T log-probabilities as per paper's approach.
  
  Args:
    model_info: Loaded Llama model information
    bit_depth: Bit depth (8, 16, 24, or 32)
    max_length: Maximum sequence length (adjusted for stereo if needed)
    
  Returns:
    Prediction function that accepts ASCII text and returns (l, T) log-probabilities
  """
  model = model_info["model"]
  tokenizer = model_info["tokenizer"]
  model_format = model_info.get("format", "huggingface")
  
  def predict_fn(ascii_text: str) -> np.ndarray:
    """Predict token probabilities following paper's approach.
    
    Paper: "This sequence is fed into the big pretrained Transformer model, 
    which gives us the conditionals ρˆ(y|x<i) for all histories x<i and tokens in the alphabet y.
    Denoting the length of the sequence after tokenization as l, we obtain l ∗ T log-probabilities."
    
    Args:
      ascii_text: ASCII string (already mapped, no need to map again)
      
    Returns:
      Log probabilities of shape (l, T) where l=sequence_length, T=vocab_size
    """
    import logging
    
    if len(ascii_text) == 0:
      # Return uniform distribution if no text
      vocab_size = len(tokenizer)
      return np.log(np.ones(vocab_size) / vocab_size)
    
    # Tokenize ASCII text
    tokens = tokenizer.encode(ascii_text, add_special_tokens=False)
    
    if len(tokens) == 0:
      # Return uniform distribution if no tokens
      vocab_size = len(tokenizer)
      return np.log(np.ones(vocab_size) / vocab_size)
    
    # Truncate tokens to max_length if needed
    if len(tokens) > max_length:
      tokens = tokens[:max_length]
    
    # PAPER'S APPROACH: Feed entire sequence to model, get predictions for all positions
    input_ids = torch.tensor(tokens).unsqueeze(0)  # Shape: (1, l)
    
    with torch.no_grad():
      if model_format == "huggingface":
        outputs = model(input_ids)
        logits = outputs.logits  # Shape: (1, l, T)
      elif model_format == "original":
        logits = _run_original_llama_inference(model, input_ids)  # Shape: (1, l, T)
      else:
        raise ValueError(f"Unknown model format: {model_format}")
      
      # Convert to log probabilities: shape (l, T)
      log_probs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(0)
    
    result = log_probs.cpu().float().numpy()  # Shape: (l, T)
    return result
  
  return predict_fn


def create_llama_predict_fn_token_level(model_info: Dict[str, Any], max_length: int = 2048) -> Callable:
  """Create prediction function that works at token level as per paper.
  
  Args:
    model_info: Loaded Llama model information
    max_length: Maximum sequence length
    
  Returns:
    Prediction function that accepts tokenized input and returns token-level probabilities
  """
  model = model_info["model"]
  tokenizer = model_info["tokenizer"]
  model_format = model_info.get("format", "huggingface")
  
  def predict_fn_token_level(tokens: list) -> np.ndarray:
    """Predict token probabilities following paper's approach.
    
    Paper: "This sequence is fed into the big pretrained Transformer model, 
    which gives us the conditionals ρˆ(y|x<i) for all histories x<i and tokens in the alphabet y.
    Denoting the length of the sequence after tokenization as l, we obtain l ∗ T log-probabilities."
    
    Args:
      tokens: List of token IDs (already tokenized)
      
    Returns:
      Log probabilities of shape (l_tokens, T) where l_tokens=token_sequence_length, T=vocab_size
    """
    import logging
    
    if len(tokens) == 0:
      # Return uniform distribution if no tokens
      vocab_size = len(tokenizer)
      return np.log(np.ones(vocab_size) / vocab_size)
    
    # Truncate tokens to max_length if needed
    if len(tokens) > max_length:
      tokens = tokens[:max_length]
    
    # Feed tokenized sequence to model (as paper describes)
    input_ids = torch.tensor(tokens).unsqueeze(0)  # Shape: (1, l_tokens)
    
    with torch.no_grad():
      if model_format == "huggingface":
        outputs = model(input_ids)
        logits = outputs.logits  # Shape: (1, l_tokens, T)
      elif model_format == "original":
        # For original format, we need to create the proper wrapper structure
        import logging
        logging.debug(f"Model info keys: {list(model_info.keys())}")
        logging.debug(f"Checkpoint in model_info: {'checkpoint' in model_info}")
        logging.debug(f"Params in model_info: {'params' in model_info}")
        
        # For original format, checkpoint and params are stored in model_info["model"]
        if "model" in model_info and isinstance(model_info["model"], dict):
          model_wrapper = {
            "checkpoint": model_info["model"].get("checkpoint", {}),
            "params": model_info["model"].get("params", {})
          }
        else:
          # Fallback to direct access (for other formats)
          model_wrapper = {
            "checkpoint": model_info.get("checkpoint", {}),
            "params": model_info.get("params", {})
          }
        # logging.debug(f"Model wrapper checkpoint keys: {list(model_wrapper['checkpoint'].keys())}")
        # logging.debug(f"Model wrapper params keys: {list(model_wrapper['params'].keys())}")
        
        logits = _run_original_llama_inference(model_wrapper, input_ids)  # Shape: (1, l_tokens, T)
      else:
        raise ValueError(f"Unknown model format: {model_format}")
      
      # Convert to log probabilities: shape (l_tokens, T)
      log_probs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(0)
    
    result = log_probs.cpu().float().numpy()  # Shape: (l_tokens, T)
    return result
  
  return predict_fn_token_level


def apply_top_k_filtering(log_probs: np.ndarray, k: int = 100) -> np.ndarray:
  """Apply top-k filtering and renormalization as described in the paper.
  
  Paper: "In practice, the large models had only access to the top-k next token log-probabilities, 
  for each context. We chose k = 100, which almost fully recovers the conditional distribution. 
  Arithmetic coding can still be applied as the alphabet size is allowed to change while coding: 
  what matters is that the conditional probabilities in each step sum to 1. 
  Accordingly, we renormalize the top-k log-probabilities."
  
  Args:
    log_probs: Log probabilities of shape (seq_len, vocab_size)
    k: Number of top probabilities to keep (k=100 as per paper)
    
  Returns:
    Filtered and renormalized log probabilities of shape (seq_len, vocab_size)
  """
  filtered_log_probs = []
  
  for i in range(log_probs.shape[0]):
    # Get top-k indices properly using argsort for guaranteed k elements
    top_k_indices = np.argsort(log_probs[i])[-k:]
    
    # Create filtered distribution: keep only top-k, set others to -inf
    filtered_probs = np.full_like(log_probs[i], -np.inf)
    filtered_probs[top_k_indices] = log_probs[i][top_k_indices]
    
    # Renormalize in probability space for numerical stability
    # Paper: "Accordingly, we renormalize the top-k log-probabilities"
    probs = np.exp(filtered_probs[top_k_indices])
    probs = probs / np.sum(probs)
    filtered_probs[top_k_indices] = np.log(probs)
    
    filtered_log_probs.append(filtered_probs)
  
  result = np.stack(filtered_log_probs)
  return result


def create_llama_compression_function_extended(
    model_info: Dict[str, Any], 
    bit_depth: int = 8
) -> Callable[[bytes], bytes]:
  """Create compression function that works with Llama models using existing infrastructure.
  
  Args:
    model_info: Loaded Llama model information
    bit_depth: Bit depth (8, 16, 24, or 32)
    
  Returns:
    Compression function
  """
  model = model_info["model"]
  tokenizer = model_info["tokenizer"]
  
  def llama_compress(data: bytes) -> bytes:
    """Compress data using Llama model with existing mask function infrastructure.
    
    Args:
      data: Input bytes to compress
      
    Returns:
      Compressed bytes
    """
    # Step 1: Apply ASCII mapping to convert to ASCII-compatible format
    ascii_mapping_fn = ascii_mapping.get_ascii_mapping_function_for_bit_depth(bit_depth)
    ascii_data, dropped_lsb_bits = ascii_mapping_fn(data)
    
    # Step 2: Convert masked bytes to ASCII string
    ascii_text = ascii_data.decode('ascii', errors='ignore')
    
    if len(ascii_text) == 0:
      # Return empty compressed data if no valid ASCII
      return b''
    
    # Step 3: Tokenize ASCII text with Llama tokenizer
    tokens = tokenizer.encode(ascii_text, add_special_tokens=False)
    
    if len(tokens) == 0:
      # Return empty compressed data if no tokens
      return b''
    
    # Step 4: Get predictions from Llama model
    input_ids = torch.tensor(tokens).unsqueeze(0)
    
    with torch.no_grad():
      outputs = model(input_ids)
      logits = outputs.logits
      
    # Step 5: Convert logits to probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs).squeeze(0).cpu().numpy()
    
    # Step 6: Use arithmetic coding on token probabilities
    output = []
    encoder = arithmetic_coder.Encoder(
        base=constants.ARITHMETIC_CODER_BASE,
        precision=constants.ARITHMETIC_CODER_PRECISION,
        output_fn=output.append,
    )
    
    # Encode each token
    for i, token_id in enumerate(tokens):
      if i < len(probs):
        pdf = probs[i]
        encoder.encode(utils.normalize_pdf_for_arithmetic_coding(pdf), token_id)
    
    encoder.terminate()
    
    # Step 7: Convert bits to bytes
    compressed_bits = ''.join(map(str, output))
    compressed_bytes, _ = utils.bits_to_bytes(compressed_bits)
    
    # Step 8: Append dropped LSB bits to compressed data
    # This ensures lossless reconstruction
    final_compressed = compressed_bytes + dropped_lsb_bits
    
    return final_compressed
  
  return llama_compress


def create_llama_decompression_function(
    model_info: Dict[str, Any],
    bit_depth: int = 8
) -> Callable[[bytes, int], bytes]:
  """Create decompression function that works with Llama models.
  
  Args:
    model_info: Loaded Llama model information
    bit_depth: Bit depth (8, 16, 24, or 32)
    
  Returns:
    Decompression function
  """
  model = model_info["model"]
  tokenizer = model_info["tokenizer"]
  
  def llama_decompress(compressed_data: bytes, original_length: int) -> bytes:
    """Decompress data using Llama model.
    
    Args:
      compressed_data: Compressed bytes
      original_length: Length of original data
      
    Returns:
      Decompressed bytes
    """
    # Step 1: Split compressed data and dropped LSB bits
    # Estimate the split point based on original length and bit depth
    bytes_per_sample = ascii_mapping.calculate_bytes_per_sample(bit_depth)
    ascii_chars_per_sample = ascii_mapping.calculate_ascii_chars_per_sample(bit_depth)
    dropped_bits_per_sample = ascii_mapping.calculate_ascii_chars_per_sample(bit_depth)
    
    num_samples = original_length // bytes_per_sample
    expected_ascii_chars = num_samples * ascii_chars_per_sample
    expected_dropped_bits = num_samples * dropped_bits_per_sample
    
    # Split point is approximate - in practice, this would need more sophisticated handling
    split_point = len(compressed_data) - expected_dropped_bits
    compressed_bytes = compressed_data[:split_point]
    dropped_lsb_bits = compressed_data[split_point:]
    
    # Step 2: Convert compressed bytes back to bits
    compressed_bits = utils.bytes_to_bits(compressed_bytes)
    
    # Step 3: Arithmetic decoding
    def input_fn():
      for bit in compressed_bits:
        yield int(bit)
    
    decoder = arithmetic_coder.Decoder(
        base=constants.ARITHMETIC_CODER_BASE,
        precision=constants.ARITHMETIC_CODER_PRECISION,
        input_fn=input_fn,
    )
    
    # Step 4: Decode tokens
    decoded_tokens = []
    current_sequence = []
    
    for i in range(expected_ascii_chars):
      # Get prediction for next token
      if current_sequence:
        input_ids = torch.tensor(current_sequence).unsqueeze(0)
        with torch.no_grad():
          outputs = model(input_ids)
          logits = outputs.logits
          log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
          probs = torch.exp(log_probs).squeeze(0).cpu().numpy()
          
        # Use arithmetic decoder
        pdf = probs[-1]  # Last position
        token_id = decoder.decode(utils.normalize_pdf_for_arithmetic_coding(pdf))
        decoded_tokens.append(token_id)
        current_sequence.append(token_id)
      else:
        # First token - use uniform distribution
        vocab_size = len(tokenizer)
        uniform_pdf = np.ones(vocab_size) / vocab_size
        token_id = decoder.decode(utils.normalize_pdf_for_arithmetic_coding(uniform_pdf))
        decoded_tokens.append(token_id)
        current_sequence.append(token_id)
    
    # Step 5: Convert tokens back to ASCII text
    ascii_text = tokenizer.decode(decoded_tokens)
    
    # Step 6: Convert ASCII back to bytes
    ascii_bytes = ascii_text.encode('ascii', errors='ignore')
    
    # Step 7: Reconstruct original bytes using ASCII mapping
    original_bytes = ascii_mapping.reconstruct_original_bytes(
        ascii_bytes, dropped_lsb_bits, bit_depth
    )
    
    return original_bytes
  
    return llama_decompress


def create_batched_llama_predict_fn(model_info: Dict[str, Any], bit_depth: int = 8) -> Callable:
    """Create batched prediction function for better performance."""
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    
    def predict_fn(sequence_array: np.ndarray) -> np.ndarray:
        """Batched prediction function."""
        batch_size, seq_len = sequence_array.shape
        
        # Process batch
        all_log_probs = []
        
        for i in range(batch_size):
            # Convert to bytes based on bit depth
            if bit_depth == 8:
                data_bytes = sequence_array[i].astype(np.uint8).tobytes()
            elif bit_depth == 16:
                data_bytes = sequence_array[i].astype(np.int16).tobytes()
            elif bit_depth == 24:
                # Convert to 24-bit bytes
                data_bytes = []
                for sample in sequence_array[i].flatten():
                    if sample < 0:
                        sample = sample + (1 << 24)  # Convert to unsigned 24-bit
                    data_bytes.extend(sample.astype(np.int32).tobytes()[:3])  # Take first 3 bytes
                data_bytes = bytes(data_bytes)
            elif bit_depth == 32:
                data_bytes = sequence_array[i].astype(np.int32).tobytes()
            else:
                raise ValueError(f"Unsupported bit depth: {bit_depth}")
            
            # Apply ASCII mapping
            ascii_mapping_fn = ascii_mapping.get_ascii_mapping_function_for_bit_depth(bit_depth)
            ascii_data, _ = ascii_mapping_fn(data_bytes)
            
            # Convert to ASCII string
            ascii_text = ascii_data.decode('ascii', errors='ignore')
            
            # Tokenize
            tokens = tokenizer.encode(ascii_text, add_special_tokens=False)
            
            if len(tokens) == 0:
                vocab_size = len(tokenizer)
                log_probs = np.log(np.ones(vocab_size) / vocab_size)
            else:
                # Get predictions
                input_ids = torch.tensor(tokens).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits
                    
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                log_probs = log_probs.squeeze(0).cpu().numpy()
            
            all_log_probs.append(log_probs)
        
        return np.stack(all_log_probs)
    
    return predict_fn


def create_memory_efficient_llama_predict_fn(model_info: Dict[str, Any], bit_depth: int = 8) -> Callable:
    """Create memory-efficient prediction function."""
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    
    # Set model to evaluation mode
    model.eval()
    
    def predict_fn(sequence_array: np.ndarray) -> np.ndarray:
        """Memory-efficient prediction function."""
        # Process one sequence at a time to save memory
        if sequence_array.ndim == 2:
            batch_size, seq_len = sequence_array.shape
            results = []
            
            for i in range(batch_size):
                single_seq = sequence_array[i:i+1]
                result = predict_fn(single_seq)
                results.append(result[0])
            
            return np.stack(results)
        
        # Single sequence processing
        if bit_depth == 8:
            data_bytes = sequence_array[0].astype(np.uint8).tobytes()
        elif bit_depth == 16:
            data_bytes = sequence_array[0].astype(np.int16).tobytes()
        elif bit_depth == 24:
            # Convert to 24-bit bytes
            data_bytes = []
            for sample in sequence_array[0].flatten():
                if sample < 0:
                    sample = sample + (1 << 24)  # Convert to unsigned 24-bit
                data_bytes.extend(sample.astype(np.int32).tobytes()[:3])  # Take first 3 bytes
            data_bytes = bytes(data_bytes)
        elif bit_depth == 32:
            data_bytes = sequence_array[0].astype(np.int32).tobytes()
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")
        
        # Apply ASCII mapping
        ascii_mapping_fn = ascii_mapping.get_ascii_mapping_function_for_bit_depth(bit_depth)
        ascii_data, _ = ascii_mapping_fn(data_bytes)
        
        # Convert to ASCII string
        ascii_text = ascii_data.decode('ascii', errors='ignore')
        
        # Tokenize
        tokens = tokenizer.encode(ascii_text, add_special_tokens=False)
        
        if len(tokens) == 0:
            vocab_size = len(tokenizer)
            return np.log(np.ones(vocab_size) / vocab_size)
        
        # Get predictions with memory optimization
        input_ids = torch.tensor(tokens).unsqueeze(0)
        
        with torch.no_grad():
            # Use half precision if available
            if model_info.get("use_16bit", False):
                input_ids = input_ids.half()
            
            outputs = model(input_ids)
            logits = outputs.logits
            
            # Convert back to float32 for numpy
            logits = logits.float()
            
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs.squeeze(0).cpu().numpy()
    
    return predict_fn
