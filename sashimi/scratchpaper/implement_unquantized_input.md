# Implementation Plan: Unquantized Input Support for SaShiMi

## Overview

This document outlines the implementation plan to add support for unquantized floating-point audio inputs to SaShiMi while maintaining quantized outputs. The key insight is that we want to change the input range from quantized integers `[0, 2^bit_depth - 1]` to floating-point values `[-1, 1]`, but keep the output as quantized logits for the categorical model.

## Current Architecture Analysis

### Current Data Flow (Quantized)
```
Raw Audio: Continuous float values [-1, 1]
    ↓
Quantization: linear_encode() or mu_law_encode()
    ↓
Quantized Values: Integer indices [0, 255] for 8-bit or [0, 65535] for 16-bit
    ↓
Input: (batch, seq_len, 1) - Raw quantized integer indices
    ↓
Encoder: nn.Embedding(n_classes, d_model) - Converts indices to dense embeddings
    ↓
SaShiMi Backbone: (batch, seq_len, d_model) → (batch, seq_len, d_model)
    ↓
SequenceDecoder: nn.Linear(d_model, n_classes) → (batch, seq_len, n_classes)
    ↓
Output: Logits for n_classes (quantized targets)
```

### Desired Architecture (Unquantized Input)
```
Raw Audio: Continuous float values [-1, 1]
    ↓
Input: (batch, seq_len, 1) - Raw floating-point values (NO quantization)
    ↓
New Encoder: Projects float values to d_model dimensions
    ↓
SaShiMi Backbone: (batch, seq_len, d_model) → (batch, seq_len, d_model) [UNCHANGED]
    ↓
SequenceDecoder: nn.Linear(d_model, n_classes) → (batch, seq_len, n_classes) [UNCHANGED]
    ↓
Output: Logits for n_classes (quantized targets) [UNCHANGED]
```

## Key Design Principles

1. **Minimal Changes**: Only modify the data loader and encoder; keep backbone and decoder unchanged
2. **Backward Compatibility**: Existing quantized models should continue to work
3. **Configuration-Driven**: Use `quantize_input` parameter in dataset configs
4. **Consistent Output**: Always output quantized logits regardless of input type

## Files to Modify

### 1. Data Loader Changes

**File**: `sashimi/s4/src/dataloaders/audio.py`

#### Changes Required:

1. **Add `quantize_input` parameter** to `AbstractAudioDataset.__init__()`:
```python
def __init__(
    self,
    bits=8,
    sample_len=None,
    quantization='linear',
    quantize_input=True,  # NEW: Whether to quantize inputs
    return_type='autoregressive',
    drop_last=True,
    target_sr=None,
    context_len=None,
    pad_len=None,
    is_stereo=False,
    interleaving_strategy='temporal',
    **kwargs,
) -> None:
```

2. **Modify `__getitem__` method** to conditionally apply quantization:
```python
def __getitem__(self, index):
    # ... existing code for loading audio ...
    
    # Apply quantization only if quantize_input=True
    if self.quantize_input:
        # Quantized signal
        qseq = self.quantizer(seq, self.bits)
        # Squeeze back to (L, 1) for mono or (L*2, 1) for stereo
        qseq = qseq.squeeze(0)
    else:
        # Keep as floating-point values
        qseq = seq.squeeze(0)  # Remove batch dimension
    
    # Return the signal
    if self.return_type == 'autoregressive':
        # ... existing code ...
```

3. **Update `QuantizedAutoregressiveAudio.init_defaults`**:
```python
@property
def init_defaults(self):
    return {
        'path': None,
        'bits': 8,
        'sample_len': None,
        'train_percentage': 0.88,
        'quantization': 'linear',
        'quantize_input': True,  # NEW: Default to True for backward compatibility
        'drop_last': False,
        'context_len': None,
        'pad_len': None,
        'output_head': 'categorical',
        'n_mixtures': 10,
        'is_stereo': False,
        'interleaving_strategy': 'temporal',
    }
```

### 2. New Encoder Implementation

**File**: `sashimi/s4/src/tasks/encoders.py`

#### Add New Encoder Class:

```python
class LinearProjectionEncoder(Encoder):
    """Linear projection encoder for unquantized floating-point inputs.
    
    This encoder projects floating-point audio values directly to d_model dimensions
    without requiring quantization or embedding lookup.
    """
    
    def __init__(self, d_input, d_model):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.projection = nn.Linear(d_input, d_model)
        
        # Initialize weights
        nn.init.normal_(self.projection.weight, mean=0, std=d_model**-.5)
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_input) where d_input=1 for mono audio
        # Output shape: (batch, seq_len, d_model)
        projected_x = self.projection(x)
        return projected_x, {}
```

#### Update Encoder Registry:

```python
registry = {
    "stop": Encoder,
    "id": nn.Identity,
    "embedding": EmbeddingWithSqueeze,
    "embedding_squeeze": EmbeddingWithSqueeze,
    "linear": nn.Linear,
    "linear_projection": LinearProjectionEncoder,  # NEW
    "position": PositionalEncoder,
    # ... rest of registry ...
}

dataset_attrs = {
    "embedding": ["n_tokens"],
    "embedding_squeeze": ["n_tokens"],
    "linear_projection": ["d_input"],  # NEW
    # ... rest of dataset_attrs ...
}

model_attrs = {
    "embedding": ["d_model"],
    "embedding_squeeze": ["d_model"],
    "linear_projection": ["d_model"],  # NEW
    # ... rest of model_attrs ...
}
```

### 3. Experiment Configuration Updates

**Files**: `sashimi/s4/configs/experiment/audio/*.yaml`

#### Create new experiment configs for unquantized inputs:

**Create `sashimi-musdb18mono-raw.yaml`:**
```yaml
# @package _global_
defaults:
  - /trainer: default
  - /loader: default
  - /dataset: musdb18mono
  - /task: multiclass_classification
  - /optimizer: adamw
  - /scheduler: plateau
  - /model: sashimi

model:
  n_layers: 8
  dropout: 0.0

train:
  monitor: val/loss
  mode: min
  train_chunk_size: 4096     # Enable chunked training with 4096 batches per chunk
  val_chunk_size: 512        # Enable chunked validation with 512 batches per chunk

task:
  metrics:
    - bpb
    - accuracy
    - accuracy@3
    - accuracy@5
    - accuracy@10

encoder: linear_projection

decoder:
  _name_: sequence
  mode: last

loader:
  batch_size: 1

trainer:
  max_epochs: 1000

optimizer:
  lr: 0.004

# Override dataset settings for unquantized inputs
dataset:
  quantize_input: false
```

**Create `sashimi-musdb18stereo-raw.yaml`:**
```yaml
# @package _global_
defaults:
  - /trainer: default
  - /loader: default
  - /dataset: musdb18stereo
  - /task: multiclass_classification
  - /optimizer: adamw
  - /scheduler: plateau
  - /model: sashimi

model:
  n_layers: 8
  dropout: 0.0

train:
  monitor: val/loss
  mode: min
  train_chunk_size: 4096     # Enable chunked training with 4096 batches per chunk
  val_chunk_size: 512        # Enable chunked validation with 512 batches per chunk

task:
  metrics:
    - bpb
    - accuracy
    - accuracy@3
    - accuracy@5
    - accuracy@10

encoder: linear_projection

decoder:
  _name_: sequence
  mode: last

loader:
  batch_size: 1

trainer:
  max_epochs: 1000
  
optimizer:
  lr: 0.004

# Override dataset settings for unquantized inputs
dataset:
  quantize_input: false
```

## Implementation Steps

### Step 1: Modify Data Loader
1. Add `quantize_input` parameter to `AbstractAudioDataset`
2. Update `__getitem__` method to conditionally apply quantization
3. Update `init_defaults` in `QuantizedAutoregressiveAudio`
4. Test with existing quantized models to ensure backward compatibility

### Step 2: Implement New Encoder
1. Create `LinearProjectionEncoder` class
2. Add to encoder registry with proper attributes
3. Test encoder with simple inputs

### Step 3: Create New Experiment Configs
1. Create `sashimi-musdb18mono-raw.yaml` with `encoder: linear_projection` and `dataset.quantize_input: false`
2. Create `sashimi-musdb18stereo-raw.yaml` with `encoder: linear_projection` and `dataset.quantize_input: false`
3. Test end-to-end with unquantized inputs

### Step 4: Validation and Testing
1. Verify parameter counts remain consistent
2. Test training with both quantized and unquantized inputs
3. Ensure output logits are identical in shape and meaning
4. Validate that existing models continue to work

## Expected Benefits

### 1. **Reduced Information Loss**
- No quantization artifacts in input processing
- Preserves full precision of audio data
- May improve model performance on high-quality audio

### 2. **Simplified Architecture**
- Eliminates embedding lookup for inputs
- Direct projection from audio values to model dimensions
- Potentially faster forward pass

### 3. **Flexibility**
- Can experiment with different input representations
- Easy to switch between quantized and unquantized inputs
- Maintains backward compatibility

## Parameter Count Impact

### Current (Quantized Input)
- **8-bit**: ~4M parameters
  - Encoder: `nn.Embedding(256, 64)` = 16,384 params
  - Decoder: `nn.Linear(64, 256)` = 16,384 params
  - Backbone: ~4M params

### New (Unquantized Input)
- **8-bit**: ~4M parameters (unchanged)
  - Encoder: `nn.Linear(1, 64)` = 64 params
  - Decoder: `nn.Linear(64, 256)` = 16,384 params
  - Backbone: ~4M params

**Net change**: -16,320 parameters (encoder becomes much smaller)

## Potential Challenges

### 1. **Input Normalization**
- Need to ensure audio values are properly normalized to [-1, 1]
- May need additional preprocessing steps

### 2. **Training Stability**
- Linear projection may require different initialization
- May need to adjust learning rates or training procedures

### 3. **Memory Usage**
- Floating-point inputs use more memory than integer indices
- May need to adjust batch sizes

## Testing Strategy

### 1. **Unit Tests**
- Test new encoder with various input shapes
- Verify data loader behavior with `quantize_input=False`
- Check parameter counts match expectations

### 2. **Integration Tests**
- Train small models with both input types
- Compare training curves and final performance
- Verify output logits have correct shapes

### 3. **Regression Tests**
- Ensure existing quantized models still work
- Test all existing dataset configurations
- Verify no breaking changes to API

## Future Extensions

### 1. **Advanced Encoders**
- Could implement more sophisticated encoders (e.g., 1D CNN)
- Support for different input representations (e.g., mel-spectrograms)

### 2. **Hybrid Approaches**
- Could support both quantized and unquantized inputs in same model
- Dynamic switching based on input type

### 3. **Performance Optimization**
- Could optimize linear projection for better performance
- Implement specialized kernels for audio processing

## Conclusion

This implementation plan provides a clean, backward-compatible way to add unquantized input support to SaShiMi. The key insight is to modify only the data loader and encoder while keeping the backbone and decoder unchanged, ensuring that the output remains quantized logits regardless of input type.

The implementation is straightforward and maintains full backward compatibility while providing the flexibility to experiment with unquantized inputs. The parameter count reduction in the encoder is a bonus, making the model slightly more efficient.
