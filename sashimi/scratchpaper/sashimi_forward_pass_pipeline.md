# Sashimi Forward Pass Pipeline: Tensor Shapes for 16-bit Categorical Model

## Overview
This document details the tensor shapes at each step of the forward pass for a 16-bit categorical Sashimi model during training.

## Input Format (from Dataset)
- **Raw input**: `(batch_size, sequence_length, 1)` - 16-bit quantized audio samples
- For 16-bit audio: `n_classes = 2^16 = 65,536` possible quantized values
- Input values are integers in range `[0, 65535]`

## Step 1: Encoder Processing
Looking at the forward pass in `train.py` lines 483-499:

```python
x, w = self.encoder(x, **z) # w can model-specific constructions
```

**Encoder**: Typically uses `EmbeddingWithSqueeze` (from configs)
- **Input**: `(batch_size, sequence_length, 1)` 
- **After squeeze**: `(batch_size, sequence_length)` - removes last dimension
- **After embedding**: `(batch_size, sequence_length, d_model)` - projects to model dimension
- **Output**: `(batch_size, sequence_length, d_model)` where `d_model` is typically 64-256

## Step 2: Sashimi Backbone Processing
```python
x, state = self.model(x, apply_output_head=False, **w, state=self._state)
```

**Sashimi Backbone** (from `sashimi.py`):
- **Input**: `(batch_size, sequence_length, d_model)`
- **Transpose**: `(batch_size, d_model, sequence_length)` - if `transposed=True` (default)
- **Down blocks**: Progressive downsampling with pooling (typically 4x4 pooling)
  - After first pooling: `(batch_size, d_model, sequence_length/4)`
  - After second pooling: `(batch_size, d_model, sequence_length/16)`
- **Center blocks**: Process at lowest resolution
- **Up blocks**: Progressive upsampling with skip connections
  - Restores original sequence length: `(batch_size, d_model, sequence_length)`
- **Transpose back**: `(batch_size, sequence_length, d_model)` - if `transposed=True`
- **Output**: `(batch_size, sequence_length, d_model)`

## Step 3: Output Head Processing
```python
if apply_output_head:
    x = self.output_head(x)
```

**Categorical Output Head** (from `output_heads.py`):
- **Input**: `(batch_size, sequence_length, d_model)`
- **Linear projection**: `nn.Linear(d_model, n_classes)`
- **Output**: `(batch_size, sequence_length, n_classes)` where `n_classes = 65,536` for 16-bit

## Step 4: Decoder Processing
```python
x, w = self.decoder(x, state=state, **z)
```

**Sequence Decoder** (from `decoders.py`):
- **Input**: `(batch_size, sequence_length, n_classes)`
- **Mode**: Typically "last" (takes last timestep) or "full" (keeps all timesteps)
- **Output**: 
  - If `mode="last"`: `(batch_size, n_classes)` - single prediction
  - If `mode="full"`: `(batch_size, sequence_length, n_classes)` - sequence of predictions

## Summary of Tensor Shapes

| Step | Component | Input Shape | Output Shape | Notes |
|------|-----------|-------------|--------------|-------|
| 1 | Dataset | - | `(B, L, 1)` | 16-bit quantized audio |
| 2 | Encoder | `(B, L, 1)` | `(B, L, d_model)` | Embedding projection |
| 3 | Sashimi Backbone | `(B, L, d_model)` | `(B, L, d_model)` | U-Net processing |
| 4 | Output Head | `(B, L, d_model)` | `(B, L, 65536)` | Categorical logits |
| 5 | Decoder | `(B, L, 65536)` | `(B, 65536)` or `(B, L, 65536)` | Final predictions |

Where:
- `B` = batch size
- `L` = sequence length  
- `d_model` = model dimension (typically 64-256)
- `65536` = number of classes for 16-bit quantization (2^16)

## The `apply_output_head` Parameter

**Question**: What is the `apply_output_head` parameter for? Why would we not want to apply the output head?

**Answer**: The `apply_output_head` parameter controls whether the backbone applies its internal output head or not. This is crucial for understanding the architecture flexibility:

### When `apply_output_head=True` (Default)
- **Backbone output**: `(batch_size, sequence_length, n_classes)` - Final predictions
- **Decoder role**: Minimal processing (often `nn.Identity()`)
- **Use case**: When the backbone has its own output head (like Sashimi with categorical/DML heads)

### When `apply_output_head=False`
- **Backbone output**: `(batch_size, sequence_length, d_model)` - Raw features
- **Decoder role**: Applies the final projection to n_classes
- **Use case**: When using external decoders or when the backbone doesn't have an output head

### Why Skip the Output Head?

1. **External Decoder Control**: Some tasks need custom decoder logic that can't be handled by the backbone's output head
2. **Weight Tying**: For language modeling, you might want to tie encoder and decoder weights
3. **Flexible Architecture**: Allows the same backbone to be used with different output strategies
4. **Feature Extraction**: Sometimes you want raw features for analysis or transfer learning

### Code Example from LMTask:
```python
if backbone_has_categorical_output:
    # Backbone already handles output projection, use identity decoder
    self.model._apply_output_head_internally = False
    decoder = nn.Identity()
else:
    # Backbone doesn't handle output, create linear decoder
    decoder = nn.Linear(decoder_input_dim, n_tokens)
```

## Purpose of the Decoder

**Question**: If the Sashimi backbone outputs n_classes, what is the point of the decoder?

**Answer**: The decoder serves several important purposes even when the backbone already outputs the correct number of classes:

### 1. **Sequence Length Control**
The decoder controls how many timesteps are used for the final prediction:
- **"last" mode**: Only uses the final timestep → `(batch_size, n_classes)`
- **"full" mode**: Uses all timesteps → `(batch_size, sequence_length, n_classes)`
- **"pool" mode**: Pools across timesteps → `(batch_size, n_classes)`

### 2. **Task-Specific Processing**
Different tasks require different output formats:
- **Classification**: Often uses "last" mode for single prediction
- **Generation**: Uses "full" mode for autoregressive generation
- **Sequence-to-sequence**: May use custom length handling

### 3. **Additional Transformations**
The decoder can apply additional processing:
- **Linear projection**: Further transform features if needed
- **Length handling**: Handle variable sequence lengths
- **Task-specific logic**: Custom processing for specific tasks

### 4. **Flexibility and Modularity**
The decoder provides a clean interface that:
- Separates backbone processing from task-specific output handling
- Allows easy swapping of different decoder strategies
- Maintains consistent interface across different model architectures

### 5. **Loss Function Compatibility**
The decoder ensures the output format matches what the loss function expects:
- Loss functions expect specific tensor shapes
- Decoder adapts backbone output to loss function requirements
- Handles edge cases like variable lengths or padding

In essence, while the backbone produces the "raw" predictions, the decoder refines and formats these predictions according to the specific task requirements, making the system more modular and flexible.
