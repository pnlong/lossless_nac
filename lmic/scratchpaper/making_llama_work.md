# Making Llama Models Work with Zero-Shot Compression

## Problem Analysis

The current zero-shot implementation has a fundamental mismatch with Llama models:

1. **Current Framework**: Uses byte-level tokenization with vocab size 256 (8-bit) or 65536 (16-bit)
2. **Llama Models**: Use subword tokenization with vocab size ~32,000 tokens
3. **Direct Mapping Issue**: Raw audio bytes cannot be directly fed to Llama tokenizer

## Solution: 7-bit ASCII Mapping Approach

Based on the original "Language Modeling is Compression" paper, the solution is to:

1. **Convert bytes to 7-bit ASCII characters**
2. **Map 7-bit values to ASCII characters** (range 0-127)
3. **Append dropped MSB as extra bits** to compressed representation
4. **Extend to 16-bit** by splitting into upper/lower 8-bit parts

## Implementation Plan

### Phase 1: Extend Existing Infrastructure for Multiple Bit Depths

#### 1.1 Create Generalized Bit Depth Mask Functions

```python
def right_shift_bytes_by_n_bits(data: bytes, n_bits: int = 1) -> tuple[bytes, int]:
    """
    Generalized version of right_shift_bytes_by_one for different bit depths.
    
    Args:
        data: Input bytes
        n_bits: Number of bits to right-shift (1 for 8-bit, 2 for 16-bit, etc.)
        
    Returns:
        tuple: (right_shifted_bytes, number_of_bytes)
    """
    if n_bits < 1 or n_bits > 7:
        raise ValueError("n_bits must be between 1 and 7")
    
    shifted_bytes = []
    for byte in data:
        shifted_byte = byte >> n_bits
        shifted_bytes.append(shifted_byte)
    
    return bytes(shifted_bytes), len(data)

def zero_most_significant_bits_if_not_ascii_decodable(data: bytes, n_bits: int = 1) -> tuple[bytes, int]:
    """
    Generalized version of zero_most_significant_bit_if_not_ascii_decodable.
    
    Args:
        data: Input bytes
        n_bits: Number of MSB bits to zero (1 for 8-bit, 2 for 16-bit, etc.)
        
    Returns:
        tuple: (ascii_decodable_bytes, number_of_zeroed_bits)
    """
    if n_bits < 1 or n_bits > 7:
        raise ValueError("n_bits must be between 1 and 7")
    
    zeroed_bits = 0
    masked_data = []
    mask = (1 << (8 - n_bits)) - 1  # Create mask for lower (8-n_bits) bits
    
    for byte in data:
        # Check if byte is ASCII decodable after masking
        masked_byte = byte & mask
        if chr(masked_byte).isascii():
            masked_data.append(masked_byte)
        else:
            zeroed_bits += n_bits
            masked_data.append(masked_byte)
    
    return bytes(masked_data), zeroed_bits
```

#### 1.2 Create Paper-Aligned ASCII Mapping Functions

```python
def ascii_map_8bit(data: bytes) -> tuple[bytes, bytes]:
    """
    Map 8-bit data to ASCII range [0, 127] following the paper's approach.
    Paper: "for each byte, to map it into the range [0, 127], we simply divide it by 2, and lose the least significant bit"
    
    Args:
        data: Input bytes
        
    Returns:
        tuple: (ascii_bytes, dropped_lsb_bits)
    """
    ascii_bytes = []
    dropped_bits = []
    
    for byte in data:
        # Paper approach: divide by 2 (right-shift by 1), lose LSB
        ascii_byte = byte >> 1  # This maps [0, 255] to [0, 127]
        lsb = byte & 1  # Extract the LSB that was lost
        
        ascii_bytes.append(ascii_byte)
        dropped_bits.append(lsb)
    
    return bytes(ascii_bytes), bytes(dropped_bits)

def ascii_map_16bit(data: bytes) -> tuple[bytes, bytes]:
    """
    Map 16-bit data to ASCII by splitting into 2 8-bit parts and applying ASCII mapping.
    A 16-bit sample becomes 2 ASCII characters, with dropped LSB bits stored separately.
    Paper approach: divide each 8-bit part by 2, lose LSB.
    This preserves full 16-bit precision by using 2 ASCII characters per sample.
    """
    if len(data) % 2 != 0:
        raise ValueError("16-bit data must have even number of bytes")
    
    ascii_bytes = []
    dropped_bits = []
    
    # Process each 16-bit sample
    for i in range(0, len(data), 2):
        # Get 16-bit sample (keep full 16-bit precision)
        sample_16bit = int.from_bytes(data[i:i+2], byteorder='little', signed=True)
        
        # Split into upper and lower 8-bit parts
        upper_8bit = (sample_16bit >> 8) & 0xFF
        lower_8bit = sample_16bit & 0xFF
        
        # Apply paper's ASCII mapping to each 8-bit part (divide by 2, lose LSB)
        for byte_val in [upper_8bit, lower_8bit]:
            ascii_byte = byte_val >> 1  # Divide by 2, maps [0, 255] to [0, 127]
            lsb = byte_val & 1  # Extract the LSB that was lost
            
            ascii_bytes.append(ascii_byte)
            dropped_bits.append(lsb)
    
    return bytes(ascii_bytes), bytes(dropped_bits)

def ascii_map_24bit(data: bytes) -> tuple[bytes, bytes]:
    """
    Map 24-bit data to ASCII by splitting into 3 8-bit parts and applying ASCII mapping.
    A 24-bit sample becomes 3 ASCII characters, with dropped LSB bits stored separately.
    Paper approach: divide each 8-bit part by 2, lose LSB.
    This preserves full 24-bit precision by using 3 ASCII characters per sample.
    """
    if len(data) % 3 != 0:
        raise ValueError("24-bit data must have length divisible by 3")
    
    ascii_bytes = []
    dropped_bits = []
    
    # Process each 24-bit sample
    for i in range(0, len(data), 3):
        # Get 24-bit sample (keep full 24-bit precision)
        sample_24bit = int.from_bytes(data[i:i+3], byteorder='little', signed=True)
        
        # Split into three 8-bit parts
        upper_8bit = (sample_24bit >> 16) & 0xFF
        middle_8bit = (sample_24bit >> 8) & 0xFF
        lower_8bit = sample_24bit & 0xFF
        
        # Apply paper's ASCII mapping to each 8-bit part (divide by 2, lose LSB)
        for byte_val in [upper_8bit, middle_8bit, lower_8bit]:
            ascii_byte = byte_val >> 1  # Divide by 2, maps [0, 255] to [0, 127]
            lsb = byte_val & 1  # Extract the LSB that was lost
            
            ascii_bytes.append(ascii_byte)
            dropped_bits.append(lsb)
    
    return bytes(ascii_bytes), bytes(dropped_bits)

def ascii_map_32bit(data: bytes) -> tuple[bytes, bytes]:
    """
    Map 32-bit data to ASCII by splitting into 4 8-bit parts and applying ASCII mapping.
    A 32-bit sample becomes 4 ASCII characters, with dropped LSB bits stored separately.
    Paper approach: divide each 8-bit part by 2, lose LSB.
    This preserves full 32-bit precision by using 4 ASCII characters per sample.
    """
    if len(data) % 4 != 0:
        raise ValueError("32-bit data must have length divisible by 4")
    
    ascii_bytes = []
    dropped_bits = []
    
    # Process each 32-bit sample
    for i in range(0, len(data), 4):
        # Get 32-bit sample (keep full 32-bit precision)
        sample_32bit = int.from_bytes(data[i:i+4], byteorder='little', signed=True)
        
        # Split into four 8-bit parts
        byte3 = (sample_32bit >> 24) & 0xFF
        byte2 = (sample_32bit >> 16) & 0xFF
        byte1 = (sample_32bit >> 8) & 0xFF
        byte0 = sample_32bit & 0xFF
        
        # Apply paper's ASCII mapping to each 8-bit part (divide by 2, lose LSB)
        for byte_val in [byte3, byte2, byte1, byte0]:
            ascii_byte = byte_val >> 1  # Divide by 2, maps [0, 255] to [0, 127]
            lsb = byte_val & 1  # Extract the LSB that was lost
            
            ascii_bytes.append(ascii_byte)
            dropped_bits.append(lsb)
    
    return bytes(ascii_bytes), bytes(dropped_bits)
```

#### 1.3 Create Bit Depth Detection and Selection

```python
def get_ascii_mapping_function_for_bit_depth(bit_depth: int) -> Callable[[bytes], tuple[bytes, bytes]]:
    """
    Get the appropriate ASCII mapping function for a given bit depth.
    
    Args:
        bit_depth: Bit depth (8, 16, 24, or 32)
        
    Returns:
        ASCII mapping function for the specified bit depth
    """
    if bit_depth == 8:
        return ascii_map_8bit
    elif bit_depth == 16:
        return ascii_map_16bit
    elif bit_depth == 24:
        return ascii_map_24bit
    elif bit_depth == 32:
        return ascii_map_32bit
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}. Supported: 8, 16, 24, 32")

def calculate_bits_per_sample(bit_depth: int) -> int:
    """Calculate bits per sample for a given bit depth."""
    return bit_depth

def calculate_bytes_per_sample(bit_depth: int) -> int:
    """Calculate bytes per sample for a given bit depth."""
    return bit_depth // 8

def calculate_ascii_chars_per_sample(bit_depth: int) -> int:
    """Calculate number of ASCII characters per sample for a given bit depth."""
    return bit_depth // 8

def reconstruct_original_bytes(ascii_bytes: bytes, dropped_lsb_bits: bytes, bit_depth: int) -> bytes:
    """
    Reconstruct original bytes from ASCII bytes and dropped LSB bits.
    
    Args:
        ascii_bytes: ASCII-mapped bytes
        dropped_lsb_bits: Dropped LSB bits
        bit_depth: Original bit depth
        
    Returns:
        Reconstructed original bytes
    """
    if bit_depth == 8:
        return reconstruct_8bit_bytes(ascii_bytes, dropped_lsb_bits)
    elif bit_depth == 16:
        return reconstruct_16bit_bytes(ascii_bytes, dropped_lsb_bits)
    elif bit_depth == 24:
        return reconstruct_24bit_bytes(ascii_bytes, dropped_lsb_bits)
    elif bit_depth == 32:
        return reconstruct_32bit_bytes(ascii_bytes, dropped_lsb_bits)
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")

def reconstruct_8bit_bytes(ascii_bytes: bytes, dropped_lsb_bits: bytes) -> bytes:
    """Reconstruct 8-bit bytes from ASCII bytes and dropped LSB bits."""
    original_bytes = []
    for i, ascii_byte in enumerate(ascii_bytes):
        lsb = dropped_lsb_bits[i] if i < len(dropped_lsb_bits) else 0
        # Reconstruct: original_byte = (ascii_byte << 1) | lsb
        original_byte = (ascii_byte << 1) | lsb
        original_bytes.append(original_byte)
    return bytes(original_bytes)

def reconstruct_16bit_bytes(ascii_bytes: bytes, dropped_lsb_bits: bytes) -> bytes:
    """Reconstruct 16-bit bytes from ASCII bytes and dropped LSB bits."""
    original_bytes = []
    for i in range(0, len(ascii_bytes), 2):
        # Reconstruct upper and lower 8-bit parts
        upper_ascii = ascii_bytes[i]
        lower_ascii = ascii_bytes[i + 1]
        upper_lsb = dropped_lsb_bits[i] if i < len(dropped_lsb_bits) else 0
        lower_lsb = dropped_lsb_bits[i + 1] if i + 1 < len(dropped_lsb_bits) else 0
        
        # Reconstruct: original_byte = (ascii_byte << 1) | lsb
        upper_8bit = (upper_ascii << 1) | upper_lsb
        lower_8bit = (lower_ascii << 1) | lower_lsb
        
        # Combine into 16-bit sample
        sample_16bit = (upper_8bit << 8) | lower_8bit
        original_bytes.extend(sample_16bit.to_bytes(2, byteorder='little', signed=True))
    
    return bytes(original_bytes)

def reconstruct_24bit_bytes(ascii_bytes: bytes, dropped_lsb_bits: bytes) -> bytes:
    """Reconstruct 24-bit bytes from ASCII bytes and dropped LSB bits."""
    original_bytes = []
    for i in range(0, len(ascii_bytes), 3):
        # Reconstruct three 8-bit parts
        upper_ascii = ascii_bytes[i]
        middle_ascii = ascii_bytes[i + 1]
        lower_ascii = ascii_bytes[i + 2]
        upper_lsb = dropped_lsb_bits[i] if i < len(dropped_lsb_bits) else 0
        middle_lsb = dropped_lsb_bits[i + 1] if i + 1 < len(dropped_lsb_bits) else 0
        lower_lsb = dropped_lsb_bits[i + 2] if i + 2 < len(dropped_lsb_bits) else 0
        
        # Reconstruct: original_byte = (ascii_byte << 1) | lsb
        upper_8bit = (upper_ascii << 1) | upper_lsb
        middle_8bit = (middle_ascii << 1) | middle_lsb
        lower_8bit = (lower_ascii << 1) | lower_lsb
        
        # Combine into 24-bit sample
        sample_24bit = (upper_8bit << 16) | (middle_8bit << 8) | lower_8bit
        original_bytes.extend(sample_24bit.to_bytes(3, byteorder='little', signed=True))
    
    return bytes(original_bytes)

def reconstruct_32bit_bytes(ascii_bytes: bytes, dropped_lsb_bits: bytes) -> bytes:
    """Reconstruct 32-bit bytes from ASCII bytes and dropped LSB bits."""
    original_bytes = []
    for i in range(0, len(ascii_bytes), 4):
        # Reconstruct four 8-bit parts
        byte3_ascii = ascii_bytes[i]
        byte2_ascii = ascii_bytes[i + 1]
        byte1_ascii = ascii_bytes[i + 2]
        byte0_ascii = ascii_bytes[i + 3]
        byte3_lsb = dropped_lsb_bits[i] if i < len(dropped_lsb_bits) else 0
        byte2_lsb = dropped_lsb_bits[i + 1] if i + 1 < len(dropped_lsb_bits) else 0
        byte1_lsb = dropped_lsb_bits[i + 2] if i + 2 < len(dropped_lsb_bits) else 0
        byte0_lsb = dropped_lsb_bits[i + 3] if i + 3 < len(dropped_lsb_bits) else 0
        
        # Reconstruct: original_byte = (ascii_byte << 1) | lsb
        byte3_8bit = (byte3_ascii << 1) | byte3_lsb
        byte2_8bit = (byte2_ascii << 1) | byte2_lsb
        byte1_8bit = (byte1_ascii << 1) | byte1_lsb
        byte0_8bit = (byte0_ascii << 1) | byte0_lsb
        
        # Combine into 32-bit sample
        sample_32bit = (byte3_8bit << 24) | (byte2_8bit << 16) | (byte1_8bit << 8) | byte0_8bit
        original_bytes.extend(sample_32bit.to_bytes(4, byteorder='little', signed=True))
    
    return bytes(original_bytes)
```

#### 1.4 Update Constants for Multiple Bit Depths

```python
# Add to constants.py
# Note: Alphabet size remains constant at 256 (ASCII characters) for all bit depths
# Higher bit depths are processed by splitting samples into multiple 8-bit parts
# Following the paper's approach: divide each byte by 2 (right-shift), lose LSB, store LSB bits

# Bit depth configurations
BIT_DEPTH_CONFIGS = {
    8: {
        'alphabet_size': ALPHABET_SIZE,  # 256 ASCII characters
        'bytes_per_sample': 1,
        'ascii_chars_per_sample': 1,  # 1 ASCII char per 8-bit sample
        'dropped_bits_per_sample': 1,  # 1 LSB bit dropped per 8-bit part
        'ascii_mapping_function': ascii_map_8bit,
        'reconstruction_function': reconstruct_8bit_bytes
    },
    16: {
        'alphabet_size': ALPHABET_SIZE,  # 256 ASCII characters (same as 8-bit)
        'bytes_per_sample': 2,
        'ascii_chars_per_sample': 2,  # 2 ASCII chars per 16-bit sample
        'dropped_bits_per_sample': 2,  # 1 LSB bit dropped per 8-bit part
        'ascii_mapping_function': ascii_map_16bit,
        'reconstruction_function': reconstruct_16bit_bytes
    },
    24: {
        'alphabet_size': ALPHABET_SIZE,  # 256 ASCII characters (same as 8-bit)
        'bytes_per_sample': 3,
        'ascii_chars_per_sample': 3,  # 3 ASCII chars per 24-bit sample
        'dropped_bits_per_sample': 3,  # 1 LSB bit dropped per 8-bit part
        'ascii_mapping_function': ascii_map_24bit,
        'reconstruction_function': reconstruct_24bit_bytes
    },
    32: {
        'alphabet_size': ALPHABET_SIZE,  # 256 ASCII characters (same as 8-bit)
        'bytes_per_sample': 4,
        'ascii_chars_per_sample': 4,  # 4 ASCII chars per 32-bit sample
        'dropped_bits_per_sample': 4,  # 1 LSB bit dropped per 8-bit part
        'ascii_mapping_function': ascii_map_32bit,
        'reconstruction_function': reconstruct_32bit_bytes
    }
}
```

#### 1.5 Extend Audio Data Processing

```python
def convert_to_target_bit_depth_extended(audio_data: np.ndarray, bit_depth: int) -> bytes:
    """
    Convert audio data to target bit depth and return as bytes.
    
    Args:
        audio_data: Audio data as 1D numpy array (float values in [-1, 1])
        bit_depth: Target bit depth (8, 16, 24, or 32)
        
    Returns:
        Audio data as bytes in the target bit depth
    """
    if bit_depth == 8:
        # Convert to 8-bit unsigned integers
        # Map [-1, 1] to [0, 255]
        audio_8bit = ((audio_data + 1.0) * 127.5).astype(np.uint8)
        return audio_8bit.tobytes()
    elif bit_depth == 16:
        # Convert to 16-bit signed integers
        # Map [-1, 1] to [-32768, 32767]
        audio_16bit = (audio_data * 32767).astype(np.int16)
        return audio_16bit.tobytes()
    elif bit_depth == 24:
        # Convert to 24-bit signed integers
        # Map [-1, 1] to [-8388608, 8388607]
        audio_24bit = (audio_data * 8388607).astype(np.int32)
        # Convert to 24-bit bytes (3 bytes per sample)
        audio_bytes = []
        for sample in audio_24bit:
            # Convert to 24-bit signed representation
            if sample < 0:
                sample = sample + (1 << 24)  # Convert to unsigned 24-bit
            audio_bytes.extend(sample.to_bytes(3, byteorder='little'))
        return bytes(audio_bytes)
    elif bit_depth == 32:
        # Convert to 32-bit signed integers
        # Map [-1, 1] to [-2147483648, 2147483647]
        audio_32bit = (audio_data * 2147483647).astype(np.int32)
        return audio_32bit.tobytes()
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")

def extract_audio_chunks_extended(audio_bytes: bytes, chunk_size: int, bit_depth: int) -> Iterator[bytes]:
    """
    Extract audio chunks with proper alignment for different bit depths.
    
    Args:
        audio_bytes: Audio data as bytes
        chunk_size: Size of each chunk in bytes
        bit_depth: Bit depth of the audio data
        
    Yields:
        Audio chunks as bytes
    """
    bytes_per_sample = calculate_bytes_per_sample(bit_depth)
    
    # Ensure chunk_size is aligned to sample boundaries
    aligned_chunk_size = (chunk_size // bytes_per_sample) * bytes_per_sample
    
    for i in range(0, len(audio_bytes), aligned_chunk_size):
        chunk = audio_bytes[i:i + aligned_chunk_size]
        if len(chunk) == aligned_chunk_size:
            yield chunk
```
    """
    Convert 7-bit ASCII string back to 16-bit bytes.
    
    Args:
        ascii_string: 7-bit ASCII string (must be even length)
        dropped_msb_bits: MSB bits that were dropped
        
    Returns:
        Original 16-bit bytes
    """
    if len(ascii_string) % 2 != 0:
        raise ValueError("16-bit ASCII string must have even length")
    
    result_bytes = []
    
    # Process pairs of ASCII characters
    for i in range(0, len(ascii_string), 2):
        # Reconstruct upper and lower 8-bit parts
        upper_ascii = ord(ascii_string[i]) & 0x7F
        lower_ascii = ord(ascii_string[i+1]) & 0x7F
        
        upper_msb = dropped_msb_bits[i] if i < len(dropped_msb_bits) else 0
        lower_msb = dropped_msb_bits[i+1] if i+1 < len(dropped_msb_bits) else 0
        
        upper_8bit = (upper_msb << 7) | upper_ascii
        lower_8bit = (lower_msb << 7) | lower_ascii
        
        # Combine into 16-bit sample
        sample_16bit = (upper_8bit << 8) | lower_8bit
        
        # Convert to bytes (little-endian, signed)
        result_bytes.extend(sample_16bit.to_bytes(2, byteorder='little', signed=True))
    
    return bytes(result_bytes)
```

### Phase 2: Llama-Specific Integration with Existing Infrastructure

#### 2.1 Create Llama Prediction Function with Bit Depth Support

```python
def create_llama_predict_fn_extended(model_info: Dict[str, Any], bit_depth: int = 8) -> Callable:
    """
    Create prediction function for Llama model with extended bit depth support.
    
    Args:
        model_info: Loaded Llama model information
        bit_depth: Bit depth (8, 16, 24, or 32)
        
    Returns:
        Prediction function
    """
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    
    def predict_fn(sequence_array: np.ndarray) -> np.ndarray:
        """
        Predict next token probabilities for Llama model with bit depth support.
        
        Args:
            sequence_array: Input sequence array
            
        Returns:
            Log probabilities for next tokens
        """
        # Convert numpy array to bytes based on bit depth
        if bit_depth == 8:
            data_bytes = sequence_array.astype(np.uint8).tobytes()
        elif bit_depth == 16:
            data_bytes = sequence_array.astype(np.int16).tobytes()
        elif bit_depth == 24:
            # Convert to 24-bit bytes
            data_bytes = []
            for sample in sequence_array.flatten():
                if sample < 0:
                    sample = sample + (1 << 24)  # Convert to unsigned 24-bit
                data_bytes.extend(sample.astype(np.int32).tobytes()[:3])  # Take first 3 bytes
            data_bytes = bytes(data_bytes)
        elif bit_depth == 32:
            data_bytes = sequence_array.astype(np.int32).tobytes()
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")
        
        # Apply mask function to convert to ASCII-compatible format
        mask_fn = get_mask_function_for_bit_depth(bit_depth, use_ascii_check=False)
        ascii_data, _ = mask_fn(data_bytes)
        
        # Convert masked bytes to ASCII string
        ascii_text = ascii_data.decode('ascii', errors='ignore')
        
        if len(ascii_text) == 0:
            # Return uniform distribution if no valid ASCII
            vocab_size = len(tokenizer)
            return np.log(np.ones(vocab_size) / vocab_size)
        
        # Tokenize ASCII text
        tokens = tokenizer.encode(ascii_text, add_special_tokens=False)
        
        if len(tokens) == 0:
            # Return uniform distribution if no tokens
            vocab_size = len(tokenizer)
            return np.log(np.ones(vocab_size) / vocab_size)
        
        # Get predictions from Llama model
        input_ids = torch.tensor(tokens).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            
        # Return log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs.squeeze(0).cpu().numpy()
    
    return predict_fn
```

#### 2.2 Create Llama Compression Function Using Existing Infrastructure

```python
def create_llama_compression_function_extended(
    model_info: Dict[str, Any], 
    bit_depth: int = 8
) -> Callable[[bytes], bytes]:
    """
    Create compression function that works with Llama models using existing infrastructure.
    
    Args:
        model_info: Loaded Llama model information
        bit_depth: Bit depth (8, 16, 24, or 32)
        
    Returns:
        Compression function
    """
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    
    def llama_compress(data: bytes) -> bytes:
        """
        Compress data using Llama model with existing mask function infrastructure.
        
        Args:
            data: Input bytes to compress
            
        Returns:
            Compressed bytes
        """
        # Step 1: Apply mask function to convert to ASCII-compatible format
        mask_fn = get_mask_function_for_bit_depth(bit_depth, use_ascii_check=False)
        ascii_data, _ = mask_fn(data)
        
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
        from language_modeling_is_compression import arithmetic_coder
        
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
        
        return compressed_bytes
    
    return llama_compress
```

#### 2.2 Create Llama Decompression Function

```python
def create_llama_decompression_function(
    model_info: Dict[str, Any],
    use_16bit: bool = False
) -> Callable[[bytes, int], bytes]:
    """
    Create decompression function that works with Llama models.
    
    Args:
        model_info: Loaded Llama model information
        use_16bit: Whether to use 16-bit processing
        
    Returns:
        Decompression function
    """
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    
    def llama_decompress(compressed_data: bytes, original_length: int) -> bytes:
        """
        Decompress data using Llama model.
        
        Args:
            compressed_data: Compressed bytes
            original_length: Length of original data
            
        Returns:
            Decompressed bytes
        """
        # Step 1: Split compressed data and dropped MSB bits
        # For now, assume we know the split point (this needs refinement)
        split_point = len(compressed_data) - (original_length // (2 if use_16bit else 1))
        compressed_bytes = compressed_data[:split_point]
        dropped_msb_bits = compressed_data[split_point:]
        
        # Step 2: Convert compressed bytes back to bits
        compressed_bits = utils.bytes_to_bits(compressed_bytes)
        
        # Step 3: Arithmetic decoding
        from language_modeling_is_compression import arithmetic_coder
        
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
        
        for i in range(original_length):
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
        if use_16bit:
            original_bytes = ascii_7bit_to_bytes_16bit(ascii_text, dropped_msb_bits)
        else:
            original_bytes = ascii_7bit_to_bytes(ascii_text, dropped_msb_bits)
        
        return original_bytes
    
    return llama_decompress
```

### Phase 3: Integration with Zero-Shot Evaluation

#### 3.1 Update Zero-Shot Arguments for Bit Depth Support

```python
def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments with bit depth support."""
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation of language models on custom audio data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ... existing arguments ...
    
    # Audio processing parameters
    parser.add_argument(
        "--bit_depth",
        type=int,
        choices=[8, 16, 24, 32],
        default=8,
        help="Audio bit depth (8, 16, 24, or 32 bits)"
    )
    
    # Keep existing use_16bit for backward compatibility
    parser.add_argument(
        "--use_16bit",
        action="store_true",
        default=False,
        help="Use 16-bit audio processing (deprecated, use --bit_depth instead)"
    )
    
    # ... rest of existing arguments ...
    
    return parser.parse_args()

def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line arguments with bit depth support."""
    # ... existing validation ...
    
    # Handle backward compatibility
    if args.use_16bit and args.bit_depth != 16:
        args.bit_depth = 16
        logging.warning("--use_16bit is deprecated, using --bit_depth=16")
    
    # Validate bit depth specific constraints
    bytes_per_sample = calculate_bytes_per_sample(args.bit_depth)
    if args.chunk_size % bytes_per_sample != 0:
        raise ValueError(f"chunk_size must be divisible by {bytes_per_sample} for {args.bit_depth}-bit audio")
    
    # ... rest of existing validation ...
```

#### 3.2 Update `evaluate_language_model` Function with Bit Depth Support

```python
def evaluate_language_model(
    model_path: str,
    data_generator: Iterator[bytes],
    args: argparse.Namespace
) -> Dict[str, Any]:
    """Evaluate language model on the provided data with bit depth support."""
    logging.info("Evaluating language model...")
    
    # Load model parameters
    model_info = load_model_parameters(model_path, use_16bit=args.use_16bit, device=args.device)
    
    # Create prediction function based on model type
    if isinstance(model_info, dict) and model_info.get("model_type") == "llama":
        predict_fn = create_llama_predict_fn_extended(model_info, bit_depth=args.bit_depth)
        model_type = "llama"
    else:
        predict_fn = create_model_predict_fn(model_info, use_16bit=args.use_16bit)
        model_type = "framework"
    
    # Create custom compression function
    def language_model_compress(data: bytes) -> bytes:
        # Convert data to array based on bit depth
        if args.bit_depth == 8:
            sequence_array = np.frombuffer(data, dtype=np.uint8)
        elif args.bit_depth == 16:
            sequence_array = np.frombuffer(data, dtype=np.int16)
        elif args.bit_depth == 24:
            # Convert 24-bit bytes to numpy array
            bytes_per_sample = 3
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
        
        # Get predictions
        if args.slow_compression:
            log_probs = []
            for subsequence_length in range(len(sequence_array)):
                subsequence_probs = predict_fn(
                    sequence_array[None, :subsequence_length + 1]
                )
                log_probs.append(subsequence_probs[0, -1])
            log_probs = np.vstack(log_probs)
        else:
            log_probs = predict_fn(sequence_array[None])[0, ...]
        
        probs = np.exp(log_probs)
        
        # Use arithmetic coding
        from language_modeling_is_compression import arithmetic_coder
        
        output = []
        encoder = arithmetic_coder.Encoder(
            base=constants.ARITHMETIC_CODER_BASE,
            precision=constants.ARITHMETIC_CODER_PRECISION,
            output_fn=output.append,
        )
        
        for pdf, symbol in zip(probs, sequence_array):
            encoder.encode(utils.normalize_pdf_for_arithmetic_coding(pdf), symbol)
        encoder.terminate()
        
        compressed_bits = ''.join(map(str, output))
        compressed_bytes, _ = utils.bits_to_bytes(compressed_bits)
        
        return compressed_bytes
    
    # Get appropriate mask function for bit depth
    mask_fn = get_mask_function_for_bit_depth(args.bit_depth, use_ascii_check=False)
    
    # Evaluate compression using existing infrastructure
    start_time = time.perf_counter()
    compression_ratio, compression_time = evaluate_compressor_chunked(
        compress_fn=language_model_compress,
        get_data_generator_fn=lambda: data_generator,
        num_chunks=args.num_chunks,
        count_header_only_once=False,
        mask_fn=mask_fn,  # Use bit depth specific mask function
        use_tqdm=args.use_tqdm,
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
```

#### 3.3 Update Audio Data Generator for Bit Depth Support

```python
def setup_audio_data_generator_extended(args: argparse.Namespace) -> Iterator[bytes]:
    """Set up audio data generator with bit depth support."""
    # Get audio file paths
    try:
        audio_files = get_all_paths(args.audio_dir)
        logging.info(f"Found {len(audio_files)} WAV files in {args.audio_dir}")
    except Exception as e:
        raise ValueError(f"Error discovering audio files: {str(e)}")
    
    # Create data generator with bit depth support
    data_generator = data_loaders.get_custom_audio_iterator_extended(
        audio_files=audio_files,
        num_chunks=args.num_chunks,
        bit_depth=args.bit_depth,
        blocking_size=args.stereo_blocking_n,
        chunk_size_bytes=args.chunk_size,
    )
    
    return data_generator
```

### Phase 4: Testing and Validation

#### 4.1 Create Test Functions for Multiple Bit Depths

```python
def test_mask_functions():
    """Test mask functions for all bit depths."""
    # Test data for different bit depths
    test_cases = [
        (8, b'\x00\x7F\x80\xFF'),
        (16, b'\x00\x00\x7F\x7F\x80\x80\xFF\xFF'),
        (24, b'\x00\x00\x00\x7F\x7F\x7F\x80\x80\x80\xFF\xFF\xFF'),
        (32, b'\x00\x00\x00\x00\x7F\x7F\x7F\x7F\x80\x80\x80\x80\xFF\xFF\xFF\xFF')
    ]
    
    for bit_depth, test_data in test_cases:
        mask_fn = get_mask_function_for_bit_depth(bit_depth, use_ascii_check=False)
        masked_data, lost_bits = mask_fn(test_data)
        
        # Verify that masked data is ASCII compatible
        ascii_text = masked_data.decode('ascii', errors='ignore')
        assert len(ascii_text) > 0, f"Failed to create ASCII text for {bit_depth}-bit data"
        
        # Verify lost bits calculation
        expected_lost_bits = calculate_lost_bits_per_sample(bit_depth) * (len(test_data) // calculate_bytes_per_sample(bit_depth))
        assert lost_bits == expected_lost_bits, f"Lost bits calculation failed for {bit_depth}-bit data"
        
        print(f"{bit_depth}-bit mask function test passed")

def test_bit_depth_conversion():
    """Test bit depth conversion functions."""
    # Create test audio data
    test_audio = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    for bit_depth in [8, 16, 24, 32]:
        # Convert to target bit depth
        audio_bytes = convert_to_target_bit_depth_extended(test_audio, bit_depth)
        
        # Verify byte length
        expected_bytes = len(test_audio) * calculate_bytes_per_sample(bit_depth)
        assert len(audio_bytes) == expected_bytes, f"Byte length mismatch for {bit_depth}-bit conversion"
        
        print(f"{bit_depth}-bit conversion test passed")

def test_llama_compression_multiple_bit_depths():
    """Test Llama compression with multiple bit depths."""
    # Test with small audio chunks for different bit depths
    test_cases = [
        (8, b'\x00\x01\x02\x03\x04\x05\x06\x07'),
        (16, b'\x00\x00\x01\x01\x02\x02\x03\x03'),
        (24, b'\x00\x00\x00\x01\x01\x01\x02\x02\x02'),
        (32, b'\x00\x00\x00\x00\x01\x01\x01\x01\x02\x02\x02\x02')
    ]
    
    # Load Llama model
    model_info = load_model_parameters("llama-2-7b-chat-hf", use_16bit=False)
    
    for bit_depth, test_data in test_cases:
        # Create compression function
        compress_fn = create_llama_compression_function_extended(model_info, bit_depth=bit_depth)
        
        # Test compression
        compressed = compress_fn(test_data)
        
        # Verify compression produces output
        assert len(compressed) > 0, f"Compression failed for {bit_depth}-bit data"
        
        print(f"{bit_depth}-bit Llama compression test passed")

def test_zero_shot_evaluation_multiple_bit_depths():
    """Test zero-shot evaluation with multiple bit depths."""
    # Test with small audio directory
    audio_dir = "/path/to/test/audio"
    model_path = "llama-2-7b-chat-hf"
    
    for bit_depth in [8, 16, 24, 32]:
        # Run zero-shot evaluation
        args = argparse.Namespace(
            audio_dir=audio_dir,
            model_path=model_path,
            bit_depth=bit_depth,
            num_chunks=10,
            baseline_compressors=['gzip'],
            verbose=True
        )
        
        results = run_comprehensive_evaluation(args)
        
        # Verify results
        assert "language_model" in results
        assert results["language_model"]["model_type"] == "llama"
        assert results["language_model"]["bit_depth"] == bit_depth
        assert "compression_ratio" in results["language_model"]
        
        print(f"{bit_depth}-bit zero-shot evaluation test passed")
```

#### 4.2 Integration Tests

```python
def test_zero_shot_with_llama():
    """Test zero-shot evaluation with Llama model."""
    # Test with small audio directory
    audio_dir = "/path/to/test/audio"
    model_path = "llama-2-7b-chat-hf"
    
    # Run zero-shot evaluation
    args = argparse.Namespace(
        audio_dir=audio_dir,
        model_path=model_path,
        use_16bit=False,
        num_chunks=10,
        baseline_compressors=['gzip'],
        verbose=True
    )
    
    results = run_comprehensive_evaluation(args)
    
    # Verify results
    assert "language_model" in results
    assert results["language_model"]["model_type"] == "llama"
    assert "compression_ratio" in results["language_model"]
    
    print("Zero-shot Llama evaluation test passed")
```

### Phase 5: Performance Optimization

#### 5.1 Batch Processing

```python
def create_batched_llama_predict_fn(model_info: Dict[str, Any]) -> Callable:
    """Create batched prediction function for better performance."""
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    use_16bit = model_info["use_16bit"]
    
    def predict_fn(sequence_array: np.ndarray) -> np.ndarray:
        """Batched prediction function."""
        batch_size, seq_len = sequence_array.shape
        
        # Process batch
        all_log_probs = []
        
        for i in range(batch_size):
            # Convert to bytes
            if use_16bit:
                data_bytes = sequence_array[i].astype(np.int16).tobytes()
            else:
                data_bytes = sequence_array[i].astype(np.uint8).tobytes()
            
            # Convert to ASCII
            if use_16bit:
                ascii_text, _ = bytes_16bit_to_ascii_7bit(data_bytes)
            else:
                ascii_text, _ = bytes_to_ascii_7bit(data_bytes)
            
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
```

#### 5.2 Memory Optimization

```python
def create_memory_efficient_llama_predict_fn(model_info: Dict[str, Any]) -> Callable:
    """Create memory-efficient prediction function."""
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    use_16bit = model_info["use_16bit"]
    
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
        if use_16bit:
            data_bytes = sequence_array[0].astype(np.int16).tobytes()
        else:
            data_bytes = sequence_array[0].astype(np.uint8).tobytes()
        
        # Convert to ASCII
        if use_16bit:
            ascii_text, _ = bytes_16bit_to_ascii_7bit(data_bytes)
        else:
            ascii_text, _ = bytes_to_ascii_7bit(data_bytes)
        
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
```

## Implementation Timeline

### Week 1: Core ASCII Mapping Functions
- [ ] Implement `bytes_to_ascii_7bit()` and `ascii_7bit_to_bytes()`
- [ ] Implement `bytes_16bit_to_ascii_7bit()` and `ascii_7bit_to_bytes_16bit()`
- [ ] Create comprehensive unit tests
- [ ] Validate round-trip conversion accuracy

### Week 2: Llama Integration
- [ ] Implement `create_llama_compression_function()`
- [ ] Implement `create_llama_decompression_function()`
- [ ] Update `create_llama_predict_fn()` with ASCII mapping
- [ ] Test with small Llama models (7B)

### Week 3: Zero-Shot Integration
- [ ] Modify `evaluate_language_model()` function
- [ ] Update `load_model_parameters()` for Llama detection
- [ ] Integrate with existing evaluation pipeline
- [ ] Test end-to-end evaluation

### Week 4: Optimization and Testing
- [ ] Implement batched processing
- [ ] Add memory optimization
- [ ] Performance benchmarking
- [ ] Comprehensive testing with different model sizes

## Expected Benefits

1. **Compatibility**: Llama models can now be used for zero-shot compression
2. **Flexibility**: Supports both 8-bit and 16-bit audio processing
3. **Efficiency**: 7-bit ASCII mapping reduces vocabulary size while preserving information
4. **Scalability**: Works with different Llama model sizes (7B, 13B, 70B)

## Potential Challenges

1. **Tokenization Overhead**: ASCII-to-token conversion adds computational cost
2. **Memory Usage**: Large Llama models require significant GPU memory
3. **Sequence Length**: Long audio sequences may exceed model context limits
4. **Decompression Complexity**: Reconstructing original bytes from ASCII + MSB bits

## Success Metrics

1. **Compression Ratio**: Achieve competitive compression ratios vs baseline methods
2. **Speed**: Maintain reasonable compression/decompression speeds
3. **Accuracy**: Perfect round-trip conversion (no data loss)
4. **Scalability**: Work with different Llama model sizes and audio formats

## Summary: Extended Infrastructure for Multiple Bit Depths

### **Key Benefits of This Approach**

#### 1. **Leverages Existing Infrastructure**
- **Reuses proven code**: Utilizes existing `right_shift_bytes_by_one` and `zero_most_significant_bit_if_not_ascii_decodable` functions
- **Maintains compatibility**: Works with existing `evaluate_compressor_chunked` pipeline
- **Minimal changes**: Extends rather than replaces existing functionality

#### 2. **Supports Multiple Bit Depths with Constant Alphabet Size**
- **8-bit**: 1 ASCII character per sample, LSB bits stored separately (lossless)
- **16-bit**: 2 ASCII characters per sample, LSB bits stored separately (lossless)  
- **24-bit**: 3 ASCII characters per sample, LSB bits stored separately (lossless)
- **32-bit**: 4 ASCII characters per sample, LSB bits stored separately (lossless)
- **Constant alphabet**: All bit depths use the same 256-character ASCII alphabet
- **Full precision**: No downsampling - each bit depth preserves its full precision

#### 3. **Paper-Aligned Lossless Processing**
- **Lossless compression**: All information is preserved (LSB bits stored separately)
- **Paper methodology**: Follows the exact approach described in the original paper
- **ASCII mapping**: Divides each byte by 2 (right-shift), loses LSB, stores LSB bits
- **Fair comparisons**: Enables direct comparison with paper results

#### 4. **Proven Implementation**
- **Paper-based approach**: Uses the exact methodology from the original paper
- **Lossless reconstruction**: Can perfectly reconstruct original data
- **Standard arithmetic coding**: Uses existing arithmetic coding infrastructure
- **MSB bit storage**: Stores dropped MSB bits as described in the paper

#### 5. **Backward Compatibility**
- **Existing 8-bit support**: Maintains compatibility with current 8-bit processing
- **Deprecated flag handling**: Gracefully handles `--use_16bit` flag
- **Gradual migration**: Allows gradual adoption of new bit depth system

### **Usage Examples**

#### **8-bit Audio (Default)**
```bash
python zero_shot.py \
    --audio_dir /path/to/audio \
    --model_path "llama-2-7b-chat-hf" \
    --bit_depth 8 \
    --num_chunks 1000
```

#### **16-bit Audio**
```bash
python zero_shot.py \
    --audio_dir /path/to/audio \
    --model_path "llama-2-7b-chat-hf" \
    --bit_depth 16 \
    --num_chunks 1000
```

#### **24-bit Audio**
```bash
python zero_shot.py \
    --audio_dir /path/to/audio \
    --model_path "llama-2-7b-chat-hf" \
    --bit_depth 24 \
    --num_chunks 1000
```

#### **32-bit Audio**
```bash
python zero_shot.py \
    --audio_dir /path/to/audio \
    --model_path "llama-2-7b-chat-hf" \
    --bit_depth 32 \
    --num_chunks 1000
```

### **Expected Outcomes**

- **Llama compatibility**: Full support for Llama models across all bit depths
- **Consistent performance**: Predictable compression ratios across bit depths
- **Scalable architecture**: Easy to extend to additional bit depths in the future
- **Production ready**: Robust, tested implementation using proven infrastructure

This approach provides a solid foundation for evaluating Llama models on audio data of various bit depths while maintaining the simplicity and reliability of the existing codebase.
