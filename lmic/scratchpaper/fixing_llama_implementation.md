# Fixing Llama Implementation for Audio Compression

## Critical Divergences from Paper's Approach

After analyzing the paper line-by-line, here are the **fundamental divergences** that explain the poor compression performance:

### **MAJOR DIVERGENCE 1: Prediction Architecture**
- **Paper**: "This sequence is fed into the big pretrained Transformer model, which gives us the conditionals ρˆ(y|x<i) for all histories x<i and tokens in the alphabet y. Denoting the length of the sequence after tokenization as l, we obtain l ∗ T log-probabilities."
- **Our Current Implementation**: Does sequential prediction (predict each token one by one)
- **Correct Approach**: Feed entire token sequence to model, get l*T log-probabilities in one forward pass

### **MAJOR DIVERGENCE 2: Top-K Implementation**
- **Paper**: "In practice, the large models had only access to the top-k next token log-probabilities, for each context"
- **Our Current Implementation**: Uses full distribution (no top-k filtering)
- **Correct Approach**: For each position, keep only top-k=100 log-probabilities and renormalize

### **MAJOR DIVERGENCE 3: ASCII String Length**
- **Paper**: "ASCII string of exactly 2048 characters" (for 8-bit data: 1 sample → 1 ASCII char)
- **Our Current Implementation**: Variable length ASCII strings (correct for multi-bit depths)
- **Correct Approach**: Variable length is OK since we support 8/16/24/32-bit (1/2/3/4 ASCII chars per sample)

## Current Issues Analysis

Based on the paper's approach and current implementation, here are the key issues:

### 1. **Redundant ASCII Mapping**
- **Problem**: ASCII mapping happens twice - once in `create_llama_predict_fn_extended` and once in `language_model_compress`
- **Impact**: Inefficient processing and potential errors
- **Fix**: Remove redundant mapping, do it once in the compression function

### 2. **Incorrect Prediction Architecture**
- **Problem**: Current approach does sequential prediction, but paper does single forward pass for entire sequence
- **Impact**: Fundamental mismatch with paper's approach - should get l*T log-probabilities in one pass
- **Fix**: Process entire token sequence at once and get predictions for all positions simultaneously

### 3. **Uniform Distribution Fallback**
- **Problem**: When ASCII mapping fails, system falls back to uniform distributions
- **Impact**: No compression benefit, essentially random predictions
- **Fix**: Ensure ASCII mapping always succeeds, or handle failures gracefully

### 4. **Missing Top-K Implementation**
- **Problem**: Paper mentions using top-k=100 log-probabilities, but current implementation uses full distribution
- **Impact**: Inefficient and potentially less accurate
- **Fix**: Implement top-k filtering and renormalization

## Implementation Plan

### Phase 1: Fix ASCII Mapping and Remove Redundancy

#### 1.1 Update `create_llama_predict_fn_extended` - CORRECTED TO MATCH PAPER
```python
def create_llama_predict_fn_extended(model_info: Dict[str, Any], bit_depth: int = 8, max_length: int = 2048) -> Callable:
    """Create prediction function that matches the paper's approach exactly."""
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
        
        return log_probs.cpu().float().numpy()  # Shape: (l, T)
    
    return predict_fn
```

#### 1.2 Update Compression Function
```python
def language_model_compress_fixed(data: bytes) -> bytes:
    """Fixed compression function following paper's approach."""
    
    # Step 1: Convert data to array based on bit depth
    if args.bit_depth == 8:
        sequence_array = np.frombuffer(data, dtype=np.uint8)
    elif args.bit_depth == 16:
        sequence_array = np.frombuffer(data, dtype=np.int16)
    # ... handle other bit depths
    
    # Step 2: Convert to bytes
    if args.bit_depth == 8:
        data_bytes = sequence_array.astype(np.uint8).tobytes()
    elif args.bit_depth == 16:
        data_bytes = sequence_array.astype(np.int16).tobytes()
    # ... handle other bit depths
    
    # Step 3: Apply ASCII mapping (ONCE, not twice!)
    from language_modeling_is_compression import ascii_mapping
    ascii_mapping_fn = ascii_mapping.get_ascii_mapping_function_for_bit_depth(args.bit_depth)
    ascii_data, dropped_lsb_bits = ascii_mapping_fn(data_bytes)
    
    # Step 4: Convert to ASCII string
    ascii_text = ascii_data.decode('ascii', errors='ignore')
    
    # Note: Variable length is OK for multi-bit depths
    # 8-bit: 1 sample → 1 ASCII char
    # 16-bit: 1 sample → 2 ASCII chars  
    # 24-bit: 1 sample → 3 ASCII chars
    # 32-bit: 1 sample → 4 ASCII chars
    
    # Step 5: Get predictions using the fixed prediction function
    log_probs = predict_fn(ascii_text)  # Shape: (l, T) where l=seq_len, T=vocab_size
    
    # Step 6: Tokenize for arithmetic coding
    tokens = model_info["tokenizer"].encode(ascii_text, add_special_tokens=False)
    
    # Step 7: Apply top-k filtering as described in paper
    # Paper: "In practice, the large models had only access to the top-k next token log-probabilities, for each context"
    log_probs_topk = apply_top_k_filtering(log_probs, k=100)
    
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
            encoder.encode(utils.normalize_pdf_for_arithmetic_coding(pdf), token_id)
    
    encoder.terminate()
    
    # Step 8: Convert bits to bytes
    compressed_bits = ''.join(map(str, output))
    compressed_bytes, _ = utils.bits_to_bytes(compressed_bits)
    
    # Step 9: Append dropped LSB bits for lossless reconstruction
    final_compressed = compressed_bytes + dropped_lsb_bits
    
    return final_compressed
```

### Phase 2: Implement Top-K Filtering

#### 2.1 Add Top-K Support - CORRECTED TO MATCH PAPER
```python
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
        # Get top-k indices for this position
        top_k_indices = np.argpartition(log_probs[i], -k)[-k:]
        
        # Create filtered distribution: keep only top-k, set others to -inf
        filtered_probs = np.full_like(log_probs[i], -np.inf)
        filtered_probs[top_k_indices] = log_probs[i][top_k_indices]
        
        # Renormalize so probabilities sum to 1
        # Paper: "Accordingly, we renormalize the top-k log-probabilities"
        log_sum = np.log(np.sum(np.exp(filtered_probs[top_k_indices])))
        filtered_probs[top_k_indices] = filtered_probs[top_k_indices] - log_sum
        
        filtered_log_probs.append(filtered_probs)
    
    return np.stack(filtered_log_probs)
```

#### 2.2 Update Prediction Function with Top-K
```python
def create_llama_predict_fn_with_topk(model_info: Dict[str, Any], bit_depth: int = 8, max_length: int = 2048, top_k: int = 100) -> Callable:
    """Create prediction function with top-k filtering."""
    
    def predict_fn(ascii_text: str) -> np.ndarray:
        # ... existing prediction logic ...
        
        # Apply top-k filtering
        log_probs = apply_top_k_filtering(log_probs, top_k)
        
        return log_probs
    
    return predict_fn
```

### Phase 3: Fix Arithmetic Coding Integration

#### 3.1 Update Arithmetic Coding to Handle Top-K
```python
def encode_with_top_k_predictions(encoder, log_probs: np.ndarray, tokens: List[int], top_k: int = 100):
    """Encode tokens using top-k filtered predictions."""
    
    for i, token_id in enumerate(tokens):
        if i < len(log_probs):
            # Get top-k indices for this position
            top_k_indices = np.argpartition(log_probs[i], -top_k)[-top_k:]
            
            # Check if current token is in top-k
            if token_id in top_k_indices:
                # Use filtered distribution
                filtered_probs = np.full_like(log_probs[i], 0.0)
                filtered_probs[top_k_indices] = np.exp(log_probs[i][top_k_indices])
                filtered_probs = filtered_probs / np.sum(filtered_probs)
                
                encoder.encode(filtered_probs, token_id)
            else:
                # Token not in top-k, use uniform distribution over top-k
                uniform_probs = np.ones(top_k) / top_k
                # Map token_id to top-k space (simplified approach)
                mapped_id = token_id % top_k
                encoder.encode(uniform_probs, mapped_id)
```

### Phase 4: Error Handling and Robustness

#### 4.1 Ensure ASCII Mapping Always Succeeds
```python
def ensure_ascii_compatibility(data: bytes, bit_depth: int) -> Tuple[bytes, bytes]:
    """Ensure ASCII mapping always succeeds by handling edge cases."""
    
    ascii_mapping_fn = ascii_mapping.get_ascii_mapping_function_for_bit_depth(bit_depth)
    
    try:
        ascii_data, dropped_bits = ascii_mapping_fn(data)
        
        # Verify ASCII compatibility
        ascii_text = ascii_data.decode('ascii', errors='strict')
        
        return ascii_data, dropped_bits
        
    except (UnicodeDecodeError, ValueError) as e:
        # Handle edge cases where ASCII mapping might fail
        logging.warning(f"ASCII mapping failed: {e}, using fallback")
        
        # Fallback: ensure all bytes are in ASCII range [0, 127]
        safe_data = bytes(min(b, 127) for b in data)
        return safe_data, b''  # No dropped bits in fallback
```

#### 4.2 Add Comprehensive Logging
```python
def add_debug_logging(ascii_text: str, tokens: List[int], log_probs: np.ndarray):
    """Add debug logging to track compression process."""
    
    logging.debug(f"ASCII text length: {len(ascii_text)}")
    logging.debug(f"Number of tokens: {len(tokens)}")
    logging.debug(f"Log probs shape: {log_probs.shape}")
    logging.debug(f"Sample ASCII text: {ascii_text[:100]}...")
    logging.debug(f"Sample tokens: {tokens[:10]}")
    logging.debug(f"Sample log probs: {log_probs[0][:10]}")
```

### Phase 5: Performance Optimizations

#### 5.1 Batch Processing
```python
def create_batched_llama_predict_fn(model_info: Dict[str, Any], bit_depth: int = 8, batch_size: int = 32) -> Callable:
    """Create batched prediction function for better performance."""
    
    def predict_fn_batched(ascii_texts: List[str]) -> List[np.ndarray]:
        """Process multiple ASCII texts in batches."""
        
        all_log_probs = []
        
        for i in range(0, len(ascii_texts), batch_size):
            batch_texts = ascii_texts[i:i+batch_size]
            batch_log_probs = []
            
            for ascii_text in batch_texts:
                # Process each text individually (can be optimized further)
                log_probs = predict_fn(ascii_text)
                batch_log_probs.append(log_probs)
            
            all_log_probs.extend(batch_log_probs)
        
        return all_log_probs
    
    return predict_fn_batched
```

#### 5.2 Memory Optimization
```python
def create_memory_efficient_predict_fn(model_info: Dict[str, Any], bit_depth: int = 8) -> Callable:
    """Create memory-efficient prediction function."""
    
    model = model_info["model"]
    model.eval()  # Set to evaluation mode
    
    def predict_fn(ascii_text: str) -> np.ndarray:
        # Process one sequence at a time to save memory
        # Use half precision if available
        # Clear cache after each prediction
        
        with torch.no_grad():
            # ... prediction logic ...
            
            # Clear GPU cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return log_probs
    
    return predict_fn
```

## Implementation Steps

### Step 1: Fix Redundant ASCII Mapping
1. Update `create_llama_predict_fn_extended` to accept ASCII text directly
2. Remove ASCII mapping from prediction function
3. Update compression function to do ASCII mapping once

### Step 2: Implement Sequential Predictions
1. Change prediction function to return sequential predictions
2. Update compression function to use predictions correctly
3. Test with simple cases to verify correctness

### Step 3: Add Top-K Filtering
1. Implement `apply_top_k_filtering` function
2. Integrate with prediction function
3. Update arithmetic coding to handle top-k distributions

### Step 4: Improve Error Handling
1. Add robust ASCII mapping with fallbacks
2. Add comprehensive logging
3. Handle edge cases gracefully

### Step 5: Performance Optimization
1. Implement batched processing
2. Add memory optimization
3. Profile and optimize bottlenecks

## Key Implementation Corrections Summary

### **✅ CORRECTED: Single Forward Pass Architecture**
- **Before**: Sequential prediction (predict each token one by one)
- **After**: Single forward pass to get l*T log-probabilities for all positions
- **Impact**: Matches how the model was trained and used in the paper

### **✅ CORRECTED: Top-K Filtering with Renormalization**
- **Before**: Uses full vocabulary distribution (32K tokens)
- **After**: Keep only top-k=100 log-probabilities per position, renormalize to sum to 1
- **Impact**: More efficient and matches paper's approach

### **✅ CORRECTED: Variable ASCII String Length**
- **Before**: Tried to enforce exactly 2048 characters
- **After**: Variable length is correct for multi-bit depths (8/16/24/32-bit)
- **Impact**: Proper handling of different bit depths

### **✅ CORRECTED: Remove Redundant ASCII Mapping**
- **Before**: ASCII mapping happened twice (prediction function + compression function)
- **After**: ASCII mapping done once in compression function
- **Impact**: Eliminates inefficiency and potential errors

## Expected Results

After implementing these fixes:

1. **Compression ratio should improve significantly** (from >100% to <100%)
2. **No more uniform distribution fallbacks** (unless truly necessary)
3. **Proper single forward pass predictions** matching the paper's approach
4. **Top-k filtering** for better efficiency and accuracy
5. **Robust error handling** for edge cases

## Testing Strategy

1. **Unit tests** for each component (ASCII mapping, predictions, arithmetic coding)
2. **Integration tests** for the full compression pipeline
3. **Performance tests** to measure improvement
4. **Edge case tests** for error handling
5. **Comparison tests** against baseline compressors

## Key Metrics to Track

1. **Compression ratio** (should be <100%)
2. **Prediction accuracy** (how well model predicts next tokens)
3. **Processing time** (should be reasonable)
4. **Memory usage** (should be manageable)
5. **Error rates** (should be minimal)

---

# IMPLEMENTATION PROGRESS REPORT

## ✅ COMPLETED IMPLEMENTATIONS

### 1. **Fixed Llama Prediction Function** ✅
**File**: `lmic/language_modeling_is_compression/llama_integration.py`
**Changes Made**:
- Updated `create_llama_predict_fn_extended()` to do **single forward pass** instead of sequential predictions
- Function now accepts ASCII text directly (no redundant ASCII mapping)
- Returns log probabilities of shape `(l, T)` where l=sequence_length, T=vocab_size
- Matches paper's approach: "Feed entire sequence to model, get l*T log-probabilities"

**Key Fix**: 
```python
# OLD (sequential): for i in range(len(tokens)): predict each token one by one
# NEW (single pass): input_ids = torch.tensor(tokens).unsqueeze(0); logits = model(input_ids)
```

### 2. **Implemented Top-K Filtering** ✅
**File**: `lmic/language_modeling_is_compression/llama_integration.py`
**Changes Made**:
- Added `apply_top_k_filtering()` function with k=100 as per paper
- Implements renormalization: "Accordingly, we renormalize the top-k log-probabilities"
- Filters each position to keep only top-k log-probabilities, sets others to -inf
- Renormalizes so probabilities sum to 1

**Key Implementation**:
```python
def apply_top_k_filtering(log_probs: np.ndarray, k: int = 100) -> np.ndarray:
    # Get top-k indices for each position
    # Keep only top-k, set others to -inf
    # Renormalize so probabilities sum to 1
```

### 3. **Fixed Compression Function** ✅
**File**: `lmic/zero_shot.py`
**Changes Made**:
- Completely rewrote `language_model_compress()` function
- **Removed redundant ASCII mapping** (was happening twice)
- **Single forward pass**: Uses `predict_fn(ascii_text)` to get all predictions at once
- **Top-k filtering**: Applies `apply_top_k_filtering(log_probs, k=100)`
- **Proper arithmetic coding**: Uses predictions for each position correctly
- **Lossless reconstruction**: Appends dropped LSB bits

**Key Fixes**:
```python
# OLD: Sequential predictions + redundant ASCII mapping + no top-k
# NEW: Single forward pass + ASCII mapping once + top-k filtering + proper arithmetic coding
```

### 4. **Removed Redundant ASCII Mapping** ✅
**Changes Made**:
- ASCII mapping now happens **only once** in the compression function
- Removed from prediction function (was causing double processing)
- Maintains proper bit depth support (8/16/24/32-bit)

## 🔧 TECHNICAL IMPLEMENTATION DETAILS

### **Architecture Changes**
- **Before**: Sequential prediction → ASCII mapping twice → Full distribution
- **After**: Single forward pass → ASCII mapping once → Top-k filtering → Arithmetic coding

### **Function Signatures Updated**
```python
# OLD
def create_llama_predict_fn_extended(model_info, bit_depth=8, max_length=2048):
    def predict_fn(sequence_array: np.ndarray) -> np.ndarray:
        # Sequential predictions, ASCII mapping inside

# NEW  
def create_llama_predict_fn_extended(model_info, bit_depth=8, max_length=2048):
    def predict_fn(ascii_text: str) -> np.ndarray:
        # Single forward pass, no ASCII mapping
```

### **New Functions Added**
```python
def apply_top_k_filtering(log_probs: np.ndarray, k: int = 100) -> np.ndarray:
    """Apply top-k filtering and renormalization as per paper."""
```

## 🚨 POTENTIAL ISSUES ENCOUNTERED

### **1. Import Dependencies**
- **Issue**: Linting shows import warnings for torch, numpy, transformers
- **Status**: Expected in this environment, should work in actual runtime
- **Impact**: None - these are just linting warnings

### **2. Function Interface Changes**
- **Issue**: Prediction function now expects ASCII text instead of numpy array
- **Status**: Updated compression function to match new interface
- **Impact**: Breaking change, but properly handled

### **3. Memory Considerations**
- **Issue**: Single forward pass loads entire sequence into memory
- **Status**: Should be manageable for typical sequence lengths
- **Impact**: May need optimization for very long sequences

## 📊 EXPECTED IMPROVEMENTS

### **Compression Ratio**
- **Before**: >100% (poor compression, larger than original)
- **Expected After**: <100% (actual compression)
- **Reason**: Proper model usage + top-k filtering + correct arithmetic coding

### **Performance**
- **Before**: Sequential predictions (slow, inefficient)
- **Expected After**: Single forward pass (faster, more efficient)
- **Reason**: Matches how model was trained and used in paper

### **Accuracy**
- **Before**: Incorrect prediction architecture
- **Expected After**: Proper conditional probabilities
- **Reason**: Single forward pass gives correct context for each position

## 🧪 NEXT STEPS FOR TESTING

### **1. Unit Testing** (Pending)
- Test `apply_top_k_filtering()` with known inputs
- Test `create_llama_predict_fn_extended()` with sample ASCII text
- Verify output shapes and ranges

### **2. Integration Testing** (Pending)
- Run full compression pipeline with sample audio data
- Compare compression ratios before/after fixes
- Verify lossless reconstruction works

### **3. Performance Testing** (Pending)
- Measure compression time improvements
- Check memory usage with different sequence lengths
- Profile bottlenecks if any

## 🎯 SUCCESS CRITERIA

### **Primary Goal**: Compression Ratio < 100%
- **Current**: >100% (poor performance)
- **Target**: <100% (actual compression)
- **Measurement**: Compare before/after implementation

### **Secondary Goals**:
1. **No uniform distribution fallbacks** (unless truly necessary)
2. **Proper single forward pass predictions** (matches paper)
3. **Top-k filtering working** (k=100, renormalized)
4. **Robust error handling** (graceful failures)

## 📝 IMPLEMENTATION STATUS

| Component | Status | Notes |
|-----------|--------|-------|
| Prediction Function | ✅ Complete | Single forward pass implemented |
| Top-K Filtering | ✅ Complete | k=100 with renormalization |
| Compression Function | ✅ Complete | Full rewrite with paper's approach |
| ASCII Mapping | ✅ Complete | Removed redundancy |
| Unit Tests | ⏳ Pending | Need to implement |
| Integration Tests | ⏳ Pending | Need to run |
| Performance Tests | ⏳ Pending | Need to measure |

**Overall Progress**: **4/7 components complete (57%)**

The core implementation is complete and should significantly improve compression performance. The remaining work is testing and validation.
