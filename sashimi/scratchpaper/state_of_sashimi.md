# Current State of SaShiMi: Forward Pass Pipeline for 8-bit and 16-bit Categorical Models

## Overview

This document describes the current forward pass pipeline for SaShiMi categorical models with both 8-bit and 16-bit quantized audio data. The architecture has been simplified to eliminate the "double decoder" problem and provide a clean separation of concerns between the backbone and decoder components.

## Key Architectural Changes

### Simplified Architecture (Current State)

The previous architecture had a confusing `apply_output_head` parameter that created ambiguity about which component was responsible for the final projection from `d_model` to `n_classes`. This has been **completely removed** in favor of a clean, unambiguous design:

```
Input: (batch_size, seq_len) -> [0, n_classes-1] integer tokens
  ↓
Encoder: nn.Embedding(n_classes, d_model) -> (batch_size, seq_len, d_model)
  ↓  
SaShiMi Backbone: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model) [always d_model]
  ↓
SequenceDecoder: output_transform = nn.Linear(d_model, n_classes) -> (batch_size, seq_len, n_classes)
  ↓
Output: Logits for n_classes (full softmax)
```

**Key principles:**
- SaShiMi backbone **never** applies an output head during training
- The decoder (SequenceDecoder) **always** handles the final projection
- No `apply_output_head` parameter needed - the backbone always outputs `d_model` features
- The "output head" is just the final linear layer in the decoder

## Forward Pass Pipeline

### Step 1: Input Processing

**8-bit Audio:**
- **Raw input**: `(batch_size, sequence_length, 1)` - 8-bit quantized audio samples
- **n_classes**: `2^8 = 256` possible quantized values
- **Input values**: Integers in range `[0, 255]`

**16-bit Audio:**
- **Raw input**: `(batch_size, sequence_length, 1)` - 16-bit quantized audio samples  
- **n_classes**: `2^16 = 65,536` possible quantized values
- **Input values**: Integers in range `[0, 65535]`

### Step 2: Encoder Processing

```python
x, w = self.encoder(x, **z) # w can model-specific constructions
```

**Encoder**: Uses `EmbeddingWithSqueeze` (from configs)
- **Input**: `(batch_size, sequence_length, 1)` 
- **After squeeze**: `(batch_size, sequence_length)` - removes last dimension
- **After embedding**: `(batch_size, sequence_length, d_model)` - projects to model dimension
- **Output**: `(batch_size, sequence_length, d_model)` where `d_model = 64`

### Step 3: SaShiMi Backbone Processing

```python
x, state = self.model(x, **w, state=self._state)
```

**SaShiMi Backbone** (from `sashimi.py`):
- **Input**: `(batch_size, sequence_length, d_model)`
- **Transpose**: `(batch_size, d_model, sequence_length)` - if `transposed=True` (default)
- **Down blocks**: Progressive downsampling with pooling (typically 4x4 pooling)
  - After first pooling: `(batch_size, d_model, sequence_length/4)`
  - After second pooling: `(batch_size, d_model, sequence_length/16)`
- **Center blocks**: Process at lowest resolution
- **Up blocks**: Progressive upsampling with skip connections
  - Restores original sequence length: `(batch_size, d_model, sequence_length)`
- **Transpose back**: `(batch_size, sequence_length, d_model)` - if `transposed=True`
- **Output**: `(batch_size, sequence_length, d_model)` - **always d_model, no output head**

### Step 4: Decoder Processing

```python
x, w = self.decoder(x, state=state, **z)
```

**SequenceDecoder** (from `decoders.py`):
- **Input**: `(batch_size, sequence_length, d_model)`
- **Output transform**: `nn.Linear(d_model, n_classes)` - projects to final classes
- **Mode**: `"last"` (takes last timestep for classification)
- **Output**: `(batch_size, n_classes)` - logits for all classes

### Step 5: Loss Computation

```python
loss = F.cross_entropy(x, targets)
```

**Cross-entropy loss**:
- **Input**: `(batch_size, n_classes)` - logits from last timestep
- **Targets**: `(batch_size,)` - target indices
- **Output**: Scalar loss value

## Tensor Shape Summary

| Step | Component | 8-bit Input Shape | 8-bit Output Shape | 16-bit Input Shape | 16-bit Output Shape | Notes |
|------|-----------|-------------------|-------------------|-------------------|-------------------|-------|
| 1 | Dataset | - | `(B, L, 1)` | - | `(B, L, 1)` | Quantized audio |
| 2 | Encoder | `(B, L, 1)` | `(B, L, 64)` | `(B, L, 1)` | `(B, L, 64)` | Embedding projection |
| 3 | SaShiMi Backbone | `(B, L, 64)` | `(B, L, 64)` | `(B, L, 64)` | `(B, L, 64)` | U-Net processing |
| 4 | SequenceDecoder | `(B, L, 64)` | `(B, 256)` | `(B, L, 64)` | `(B, 65536)` | Final projection |
| 5 | Loss | `(B, 256)` | Scalar | `(B, 65536)` | Scalar | Cross-entropy |

Where:
- `B` = batch size
- `L` = sequence length  
- `d_model` = 64 (model dimension)
- `256` = number of classes for 8-bit quantization (2^8)
- `65536` = number of classes for 16-bit quantization (2^16)

## Parameter Count Breakdown

### 8-bit Categorical Model

**Total parameter count (~4M)**:
- **SaShiMi backbone**: ~4M parameters (S4 layers, U-Net architecture)
- **Encoder embedding**: `nn.Embedding(256, 64)` = 16,384 parameters
- **SequenceDecoder output_transform**: `nn.Linear(64, 256)` = 16,384 parameters  
- **Other components**: ~0.6M parameters

**Total**: ~4M parameters ✅

### 16-bit Categorical Model

**Total parameter count (~13M)**:
- **SaShiMi backbone**: ~4M parameters (S4 layers, U-Net architecture)
- **Encoder embedding**: `nn.Embedding(65536, 64)` = 4,194,304 parameters
- **SequenceDecoder output_transform**: `nn.Linear(64, 65536)` = 4,194,304 parameters
- **Other components**: ~0.6M parameters

**Total**: ~13M parameters ✅

## Key Differences from Previous Architecture

### What Was Fixed

1. **Eliminated "Double Decoder" Problem**: 
   - **Before**: SaShiMi backbone had internal output head + SequenceDecoder both projecting to n_classes
   - **After**: Only SequenceDecoder projects to n_classes, backbone always outputs d_model

2. **Removed `apply_output_head` Parameter**:
   - **Before**: Confusing parameter controlling whether backbone applies output head
   - **After**: Backbone never applies output head, decoder always handles final projection

3. **Simplified Parameter Passing**:
   - **Before**: Complex attribute extraction causing dimension mismatches
   - **After**: Clean parameter passing with correct dimensions

4. **Fixed Memory Usage**:
   - **Before**: Massive intermediate tensors `(batch, seq_len, 65536)` in backbone
   - **After**: Only final output tensor `(batch, seq_len, 65536)` in decoder

### Benefits of Current Architecture

1. **Clear separation of concerns**: 
   - SaShiMi backbone: Sequence modeling only
   - Decoder: Final projection to output classes

2. **No architectural ambiguity**: 
   - No `apply_output_head` parameter to confuse
   - Always clear which component handles what

3. **Consistent behavior**: 
   - Backbone always outputs `d_model` features
   - Decoder always handles final projection

4. **Easier to understand and maintain**: 
   - Simpler code paths
   - No conditional logic for output heads

## Configuration Examples

### 8-bit Model Configuration
```yaml
# Dataset config
dataset:
  bits: 8
  n_classes: 256

# Model config  
model:
  d_model: 64
  output_head: categorical  # Not used, kept for compatibility

# Decoder config
decoder:
  _name_: sequence
  mode: last  # For classification
```

### 16-bit Model Configuration
```yaml
# Dataset config
dataset:
  bits: 16
  n_classes: 65536

# Model config
model:
  d_model: 64
  output_head: categorical  # Not used, kept for compatibility

# Decoder config
decoder:
  _name_: sequence
  mode: last  # For classification
```

## Memory Usage

### 8-bit Model
- **Encoder embedding**: 256 × 64 = 16,384 parameters
- **Decoder output**: 64 × 256 = 16,384 parameters
- **Total output head**: 32,768 parameters
- **Memory per sample**: ~128KB for logits

### 16-bit Model  
- **Encoder embedding**: 65536 × 64 = 4,194,304 parameters
- **Decoder output**: 64 × 65536 = 4,194,304 parameters
- **Total output head**: 8,388,608 parameters
- **Memory per sample**: ~262MB for logits

## Training Considerations

### Loss Function
- **Cross-entropy loss** with full softmax over all n_classes
- **Targets**: Ground truth quantized audio values
- **Gradient flow**: Through decoder → backbone → encoder

### Optimization
- **Learning rate**: Typically 0.01 for AdamW optimizer
- **Batch size**: 1 (due to memory constraints for 16-bit models)
- **Chunked training**: Enabled for long sequences

### Evaluation Metrics
- **Bits per byte (bpb)**: Primary metric for audio generation
- **Accuracy**: Top-k accuracy for categorical predictions
- **Perplexity**: Alternative measure of model confidence

## Files Modified

1. `sashimi/s4/src/models/sequence/backbones/sashimi.py` - Removed output head logic
2. `sashimi/s4/src/tasks/tasks.py` - Always create linear decoder
3. `sashimi/s4/train.py` - Removed apply_output_head parameter
4. `sashimi/s4/src/tasks/decoders.py` - Fixed parameter passing

## Current Status

The model now trains successfully with:
- ✅ Correct parameter count (~4M for 8-bit, ~13M for 16-bit)
- ✅ Proper memory usage (no OOM errors)
- ✅ No dimension mismatches
- ✅ Clean training logs without debug output
- ✅ Simplified, maintainable architecture
