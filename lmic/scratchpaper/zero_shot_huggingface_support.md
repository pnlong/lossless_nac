# Llama Model Support Implementation Plan for zero_shot.py

## Overview

This document outlines the implementation plan to add Llama model support to `zero_shot.py`, enabling evaluation of Llama models on custom audio data using Meta's official implementation pattern.

## Current Architecture Analysis

### Existing Model Loading System
- Uses Haiku/JAX framework with `.npz` parameter files
- Expects `hk.Params` format with specific parameter structure
- Creates prediction functions using `transformer.transformer_decoder`
- Supports 8-bit and 16-bit audio processing

### Limitations
- Only supports models trained with the framework's training scripts
- Requires `.npz` format parameter files
- Limited to framework's transformer architecture
- Cannot use Llama models directly

## Implementation Strategy

### Phase 1: Dual Model Support Architecture

#### 1.1 Model Type Detection
```python
def detect_model_type(model_path: str) -> str:
    """Detect whether model is Llama or framework format."""
    if model_path.startswith("llama:"):
        return "llama"
    elif model_path.endswith(".npz"):
        return "framework"
    elif os.path.exists(model_path) and os.path.isdir(model_path):
        # Check if it's a Llama model directory
        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    if config.get("model_type") == "llama":
                        return "llama"
            except:
                pass
    elif "/" in model_path and not os.path.exists(model_path):
        # Likely a Hugging Face model name (contains slash but file doesn't exist)
        return "llama"
    else:
        raise ValueError(f"Unknown model format: {model_path}")
```

#### 1.2 Unified Model Loading Interface
```python
def load_model_parameters(model_path: str, use_16bit: bool = False, device: str = "auto") -> Any:
    """Load model parameters from either Llama or framework format."""
    model_type = detect_model_type(model_path)
    
    if model_type == "llama":
        return load_llama_model(model_path, use_16bit, device)
    else:
        return load_framework_model(model_path)
```

### Phase 2: Llama Model Integration

#### 2.1 Llama Model Loader (Following Meta's Pattern)
```python
def load_llama_model(model_dir: str, use_16bit: bool = False, device: str = "auto") -> Dict[str, Any]:
    """Load Llama model using Meta's official pattern."""
    # Remove llama: prefix if present
    if model_dir.startswith("llama:"):
        model_dir = model_dir.replace("llama:", "")
    
    logging.info(f"Loading Llama model from: {model_dir}")
    
    try:
        # Load model and tokenizer using Meta's pattern
        model = LlamaForCausalLM.from_pretrained(model_dir)
        tokenizer = LlamaTokenizer.from_pretrained(model_dir)
        
        # Create pipeline for inference
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16 if use_16bit else torch.float32,
            device_map=device,
        )
        
        logging.info(f"Successfully loaded Llama model from {model_dir}")
        return {
            "model": model,
            "tokenizer": tokenizer,
            "pipeline": pipeline,
            "model_type": "llama",
            "use_16bit": use_16bit
        }
    except Exception as e:
        raise ValueError(f"Error loading Llama model from '{model_dir}': {str(e)}")
```

#### 2.2 Llama Prediction Function
```python
def create_llama_predict_fn(model_info: Dict[str, Any]) -> Callable:
    """Create prediction function for Llama model."""
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    use_16bit = model_info["use_16bit"]
    
    def predict_fn(sequence_array: np.ndarray) -> np.ndarray:
        """Predict next token probabilities for Llama model."""
        # Convert numpy array to tokens
        if use_16bit:
            tokens = sequence_array.astype(np.int16)
        else:
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
```

### Phase 3: Evaluation Engine Modification

#### 3.1 Unified Evaluation Function
```python
def evaluate_language_model(
    model_path: str,
    data_generator: Iterator[bytes],
    args: argparse.Namespace
) -> Dict[str, Any]:
    """Evaluate language model on the provided data."""
    logging.info("Evaluating language model...")
    
    # Load model parameters
    model_info = load_model_parameters(model_path, use_16bit=args.use_16bit)
    
    # Create prediction function based on model type
    if model_info.get("model_type") == "llama":
        predict_fn = create_llama_predict_fn(model_info)
    else:
        predict_fn = create_model_predict_fn(model_info, use_16bit=args.use_16bit)
    
    # Create custom compression function
    def language_model_compress(data: bytes) -> bytes:
        # Convert data to array
        if args.use_16bit:
            sequence_array = np.frombuffer(data, dtype=np.int16)
        else:
            sequence_array = np.frombuffer(data, dtype=np.uint8)
        
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
    
    # Evaluate compression
    start_time = time.perf_counter()
    compression_ratio, compression_time = evaluate_compressor_chunked(
        compress_fn=language_model_compress,
        get_data_generator_fn=lambda: data_generator,
        num_chunks=args.num_chunks,
        count_header_only_once=False,
        mask_fn=utils.right_shift_bytes_by_one,
        use_tqdm=args.use_tqdm,
    )
    total_time = time.perf_counter() - start_time
    
    return {
        "compression_ratio": compression_ratio,
        "compression_time": compression_time,
        "total_time": total_time,
        "compressor_type": "language_model",
        "model_type": model_info.get("model_type", "framework")
    }
```

### Phase 4: Command-Line Interface Updates

#### 4.1 Enhanced Model Path Argument
```python
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to model file (.npz) or Llama model directory (e.g., './llama-2-7b-chat-hf')"
)
```

#### 4.2 Model-Specific Configuration
```python
parser.add_argument(
    "--device",
    type=str,
    default="auto",
    help="Device to use for Llama models (auto, cpu, cuda, cuda:0, etc.)"
)

parser.add_argument(
    "--max_length",
    type=int,
    default=2048,
    help="Maximum sequence length for Llama models"
)
```

### Phase 5: Error Handling and Validation

#### 5.1 Model Validation
```python
def validate_model_path(model_path: str) -> None:
    """Validate model path and provide helpful error messages."""
    if model_path.startswith("llama:"):
        model_dir = model_path.replace("llama:", "")
        if not os.path.exists(model_dir):
            raise ValueError(f"Llama model directory does not exist: {model_dir}")
        config_file = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_file):
            raise ValueError(f"Llama model directory missing config.json: {model_dir}")
    elif not os.path.exists(model_path):
        raise ValueError(f"Model file does not exist: {model_path}")
    elif not model_path.endswith(".npz"):
        raise ValueError(f"Framework model must be .npz file: {model_path}")
```

#### 5.2 Memory Management
```python
def setup_llama_model(model_dir: str, device: str = "auto") -> Dict[str, Any]:
    """Setup Llama model with memory optimization."""
    try:
        # Load model with memory optimization using Meta's pattern
        model = LlamaForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,  # Use half precision to save memory
            device_map=device,
            low_cpu_mem_usage=True
        )
        
        tokenizer = LlamaTokenizer.from_pretrained(model_dir)
        
        # Create optimized pipeline
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map=device,
        )
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "pipeline": pipeline,
            "model_type": "llama"
        }
    except Exception as e:
        logging.warning(f"Failed to load Llama model with memory optimization: {e}")
        # Fallback to standard loading
        return load_llama_model(model_dir, device=device)
```

### Phase 6: Performance Optimizations

#### 6.1 Batch Processing
```python
def create_batched_predict_fn(model_info: Dict[str, Any], batch_size: int = 32) -> Callable:
    """Create batched prediction function for better performance."""
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    
    def batched_predict_fn(sequence_array: np.ndarray) -> np.ndarray:
        """Batched prediction for better GPU utilization."""
        # Process in batches for better performance
        batch_size = min(batch_size, sequence_array.shape[0])
        results = []
        
        for i in range(0, sequence_array.shape[0], batch_size):
            batch = sequence_array[i:i+batch_size]
            input_ids = torch.tensor(batch)
            
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                results.append(log_probs.cpu().numpy())
        
        return np.concatenate(results, axis=0)
    
    return batched_predict_fn
```

#### 6.2 Caching and Optimization
```python
def optimize_huggingface_model(model: Any) -> Any:
    """Apply optimizations to Hugging Face model."""
    # Enable compilation if available
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    # Set to evaluation mode
    model.eval()
    
    # Enable attention optimization if available
    if hasattr(model, 'enable_attention_slicing'):
        model.enable_attention_slicing()
    
    return model
```

## Implementation Details

### Dependencies to Add
```python
# Add to imports
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
import warnings
import json
```

### Configuration Updates
```python
# Add to constants
DEFAULT_DEVICE = "auto"
DEFAULT_MAX_LENGTH = 2048
DEFAULT_BATCH_SIZE = 32
DEFAULT_TORCH_DTYPE = "float16"
```

### Error Handling Strategy
1. **Model Loading Errors**: Graceful fallback to CPU if GPU fails
2. **Memory Errors**: Automatic batch size reduction
3. **Tokenization Errors**: Fallback to byte-level tokenization
4. **Compatibility Issues**: Clear error messages with suggestions

## Usage Examples

### Basic Llama Usage
```bash
python zero_shot.py \
    --audio_dir /path/to/audio \
    --model_path "./llama-2-7b-chat-hf" \
    --baseline_compressors gzip flac lzma
```

### Advanced Llama Configuration
```bash
python zero_shot.py \
    --audio_dir /path/to/audio \
    --model_path "./llama-2-7b-chat-hf" \
    --use_16bit \
    --device "cuda:0" \
    --max_length 4096 \
    --baseline_compressors gzip flac lzma \
    --output_file results.json
```

### Model Size Comparison
```bash
# 7B model
python zero_shot.py --audio_dir /path/to/audio --model_path "./llama-2-7b-chat-hf" --output_file llama_7b_results.json

# 13B model  
python zero_shot.py --audio_dir /path/to/audio --model_path "./llama-2-13b-chat-hf" --output_file llama_13b_results.json

# 70B model
python zero_shot.py --audio_dir /path/to/audio --model_path "./llama-2-70b-chat-hf" --output_file llama_70b_results.json
```

## Testing Strategy

### Unit Tests
1. Model type detection
2. Hugging Face model loading
3. Prediction function creation
4. Error handling scenarios

### Integration Tests
1. End-to-end evaluation with different model types
2. Memory usage validation
3. Performance benchmarking
4. Compatibility testing across different model sizes

### Test Models
- **7B**: `./llama-2-7b-chat-hf` (7B parameters)
- **13B**: `./llama-2-13b-chat-hf` (13B parameters)
- **70B**: `./llama-2-70b-chat-hf` (70B parameters)

## Performance Considerations

### Memory Management
- Use half precision (float16) for large models
- Enable memory efficient attention
- Implement gradient checkpointing for very large models
- Automatic batch size adjustment based on available memory

### Speed Optimizations
- Enable torch compilation when available
- Use batched inference
- Cache model predictions when possible
- Optimize attention mechanisms

## Future Enhancements

### Phase 7: Advanced Features
1. **Multi-GPU Support**: Distribute large models across multiple GPUs
2. **Model Quantization**: Support for quantized models (4-bit, 8-bit)
3. **Streaming Inference**: Process audio data in real-time
4. **Custom Tokenization**: Support for custom tokenization strategies

### Phase 8: Integration Improvements
1. **Model Registry**: Centralized model configuration management
2. **Automatic Model Selection**: Choose optimal model based on hardware
3. **Performance Profiling**: Built-in performance analysis tools
4. **Result Comparison**: Automated comparison across model sizes

## Implementation Timeline

### Week 1: Core Llama Integration
- Implement model type detection
- Add Llama model loading using Meta's pattern
- Create unified prediction interface

### Week 2: Evaluation Engine Updates
- Modify evaluation functions
- Add error handling and validation
- Implement performance optimizations

### Week 3: Testing and Optimization
- Add comprehensive tests
- Optimize memory usage
- Performance benchmarking

### Week 4: Documentation and Polish
- Update documentation
- Add usage examples
- Final testing and bug fixes

This implementation plan provides a comprehensive roadmap for adding Llama support to `zero_shot.py` while maintaining backward compatibility with the existing framework.
