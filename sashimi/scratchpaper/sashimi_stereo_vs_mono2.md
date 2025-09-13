# SaShiMi Stereo vs Mono: Unified Implementation Analysis

This document traces the complete data flow from raw audio input to final model output, comparing mono and stereo processing paths in the **unified SaShiMi stereo implementation**.

## Overview

The **unified stereo implementation** treats stereo as a **pure data preprocessing problem** where both mono and stereo use the same model architecture and task processing. The key difference is in how stereo channels are interleaved at the data level before quantization.

### Key Architectural Changes from Previous Implementation

1. **Unified Task Processing**: Both mono and stereo use `LMTask` with identical embedding strategies
2. **Data-Level Interleaving**: Stereo channels are interleaved in the dataset before quantization
3. **No Special Stereo Encoders**: Removed `StereoEncoder` and `IdentityEncoder` classes
4. **Automatic Embedding Squeeze**: The `embedding` encoder automatically handles dimension squeezing

## 1. Data Loading & Preprocessing (audio.py)

### Mono Processing (`is_stereo=False`)
```python
# In AbstractAudioDataset.__getitem__()
seq, sr = torchaudio.load(file_name, frame_offset=start_frame, num_frames=num_frames)

# Average non-mono signals across channels (only for mono processing)
if seq.shape[0] > 1 and not self.is_stereo:
    seq = seq.mean(dim=0, keepdim=True)  # (channels, length) -> (1, length)

# Transpose to (length, channels) where channels=1 for mono
seq = seq.transpose(0, 1)  # (1, length) -> (length, 1)

# Add batch dimension: (1, length, 1)
seq = seq.unsqueeze(0)

# Quantize and squeeze back to (length, 1)
qseq = self.quantizer(seq, self.bits).squeeze(0)
```

**Mono Data Shape**: `(length, 1)` - Single channel audio

### Stereo Processing (`is_stereo=True`)
```python
# In AbstractAudioDataset.__getitem__()
seq, sr = torchaudio.load(file_name, frame_offset=start_frame, num_frames=num_frames)

# NO averaging - preserve both channels
# seq.shape = (2, length) for stereo

# Transpose to (length, channels) where channels=2 for stereo
seq = seq.transpose(0, 1)  # (2, length) -> (length, 2)

# NEW: Apply configurable interleaving strategy for stereo
if self.is_stereo and seq.shape[1] == 2:
    if self.interleaving_strategy == 'temporal':
        # Interleave channels: [L, R, L, R, ...]
        # Reshape from (L, 2) to (L*2, 1)
        seq = seq.view(-1, 1)  # (L*2, 1)
    elif self.interleaving_strategy.startswith('blocking'):
        if self.interleaving_strategy == 'blocking':
            # Block channels: [L, L, L, ..., R, R, R, ...]
            # Concatenate channels: (L, 2) -> (L*2, 1)
            seq = torch.cat([seq[:, 0], seq[:, 1]], dim=0).unsqueeze(-1)  # (L*2, 1)
        else: # some block size parameter is provided
            block_size = int(self.interleaving_strategy.split('-')[-1])
            if block_size <= 0: # Validate block size
                raise ValueError(f"Block size must be positive, got {block_size}")
            seq_len = seq.shape[0]
            block_size = min(block_size, seq_len) # Truncate block size if it's larger than sequence length
            interleaved_blocks = []
            for start in range(0, seq_len, block_size):
                end = min(start + block_size, seq_len)
                interleaved_blocks.append(seq[start:end].flatten())
            seq = torch.cat(interleaved_blocks, dim=0).unsqueeze(-1)  # (L*2, 1)

# Add batch dimension: (1, L*2, 1) for stereo
seq = seq.unsqueeze(0)

# Quantize and squeeze back to (L*2, 1)
qseq = self.quantizer(seq, self.bits).squeeze(0)
```

**Stereo Data Shape**: `(length * 2, 1)` - Interleaved stereo channels as single channel

### Dataset Properties
```python
# In QuantizedAutoregressiveAudio
@property
def d_input(self):
    # Always return 1 since stereo is interleaved to single channel at data level
    return 1

@property
def d_output(self):
    """Output dimension depends on output head type"""
    output_head = getattr(self, 'output_head', 'categorical')
    if output_head == "categorical":
        return 1 << self.bits  # Always calculate from bits for consistency
    elif output_head == "dml":
        n_mixtures = getattr(self, 'n_mixtures', 10)
        return 3 * n_mixtures
```

**Key Change**: `d_input` always returns `1` because stereo is interleaved to single channel at data level.

## 2. Encoder Processing (tasks.py)

### Unified Encoder Path
Both mono and stereo now use the **same encoder processing**:

```python
# In LMTask.__init__()
# Both mono and stereo use registry-based embedding with automatic squeeze
from src.tasks.encoders import registry
embedding = registry["embedding"](n_tokens, d_model)  # Uses EmbeddingWithSqueeze

encoder = U.PassthroughSequential(
    embedding,
    scale,
)
self.encoder = encoder
```

### EmbeddingWithSqueeze Implementation
```python
# In encoders.py
class EmbeddingWithSqueeze(Encoder):
    def __init__(self, n_tokens, d_model):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, d_model)
        nn.init.normal_(self.embedding.weight, mean=0, std=d_model**-.5)
    
    def forward(self, x):
        # Automatically squeeze trailing dimension of size 1
        if x.dim() == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        
        embedded_x = self.embedding(x)
        return embedded_x, {}
```

### Encoder Outputs

**Mono Encoder Output**: 
- Input: `(batch, seq_len, 1)` 
- After squeeze: `(batch, seq_len)`
- After embedding: `(batch, seq_len, d_model)`

**Stereo Encoder Output**: 
- Input: `(batch, seq_len * 2, 1)`
- After squeeze: `(batch, seq_len * 2)`
- After embedding: `(batch, seq_len * 2, d_model)`

### Key Architectural Simplification

1. **No Special Encoders**: Removed `StereoEncoder` and `IdentityEncoder` classes
2. **Unified Processing**: Both mono and stereo use identical `EmbeddingWithSqueeze`
3. **Automatic Squeeze**: The embedding automatically handles dimension squeezing
4. **Registry-Based**: Uses the encoder registry for consistent behavior

## 3. Model Configuration (train.py)

### Sequence Length Configuration
```python
# In SequenceLightningModule.setup()
if hasattr(self.dataset, 'is_stereo') and self.dataset.is_stereo:
    # For stereo, the model needs to know about the doubled sequence length
    if '__l_max' in self.hparams.dataset:
        model_config['l_max'] = self.hparams.dataset['__l_max']
    
    # Pass stereo configuration to model
    model_config['is_stereo'] = True
    model_config['interleaving_strategy'] = getattr(self.dataset, 'interleaving_strategy', 'temporal')

# Input dimension is always 1 (stereo is interleaved to single channel)
model_config['d_input'] = 1
```

### Automatic l_max Calculation
```yaml
# In musdb18stereo.yaml
sample_len: 12288
__l_max: ${eval:${.sample_len} * 2}  # Automatically double the sample_len for stereo
```

**Key Change**: `d_input` is always `1` because stereo is interleaved to single channel at data level.

## 4. Forward Pass Processing (train.py)

### Unified Forward Pass
```python
def forward(self, batch):
    """Passes a batch through the encoder, backbone, and decoder"""
    x, y, *z = batch
    
    # No special stereo handling needed - data is already interleaved
    # x shape: (batch, seq_len, 1) for mono or (batch, seq_len*2, 1) for stereo
    
    x, w = self.encoder(x, **z)  # Standard EmbeddingWithSqueeze embedding
    x, state = self.model(x, **w, state=self._state)  # Model processes interleaved sequence
    x, w = self.decoder(x, state=state, **z)
    return x, y, w
```

**Key Simplification**: No special stereo handling in forward pass - the interleaving is handled at data level.

### Model Input Dimensions

**Mono Model Input**: 
- Data: `(batch, seq_len, 1)`
- After encoder: `(batch, seq_len, d_model)`

**Stereo Model Input**: 
- Data: `(batch, seq_len * 2, 1)`
- After encoder: `(batch, seq_len * 2, d_model)`

## 5. Model Processing (sashimi.py)

### SaShiMi Model Configuration
```python
# In Sashimi.__init__()
def __init__(
    self,
    d_model,
    n_layers,
    # ... other parameters ...
    is_stereo=False,           # New: whether processing stereo audio
    interleaving_strategy='temporal',  # New: interleaving strategy for stereo
):
    # ... initialization ...
    
    # Stereo configuration
    self.is_stereo = is_stereo
    self.interleaving_strategy = interleaving_strategy
```

### Stereo Output Reshaping
```python
def reshape_stereo_output(self, x):
    """
    Reshape stereo output from interleaved format back to stereo channels.
    
    Args:
        x: (batch, seq_len*2, d_output) - interleaved stereo output
        
    Returns:
        x: (batch, seq_len, 2, d_output) - reshaped stereo output
    """
    if not self.is_stereo:
        return x
        
    batch_size, seq_len_interleaved, d_output = x.shape
    seq_len = seq_len_interleaved // 2
    
    if self.interleaving_strategy == 'temporal':
        # Temporal interleaving: [L, R, L, R, ...] -> [L, L, L, ...], [R, R, R, ...]
        x = x.view(batch_size, seq_len, 2, d_output)
    elif self.interleaving_strategy.startswith('blocking'):
        if self.interleaving_strategy == 'blocking':
            # Blocking interleaving: [L, L, L, ..., R, R, R, ...] -> [L, L, L, ...], [R, R, R, ...]
            return tensor.view(batch_size, 2, seq_len, d_output).transpose(1, 2)
        else: # some block size parameter is provided
            block_size = int(self.interleaving_strategy.split('-')[-1])
            if block_size <= 0: # Validate block size
                raise ValueError(f"Block size must be positive, got {block_size}")
            block_size = min(block_size, seq_len) # Truncate block size if it's larger than sequence length                
            complete_blocks = seq_len // block_size # Calculate actual number of complete blocks and remainder
            remainder = seq_len % block_size # Calculate actual number of complete blocks and remainder
            if remainder == 0: # All blocks are complete
                reshaped_tensor = tensor.view(batch_size, complete_blocks, 2, block_size, d_output)
                return reshaped_tensor.view(batch_size, -1, 2, d_output)
            else: # Handle variable block sizes
                complete_tensor = tensor[:, :complete_blocks * 2 * block_size, :]
                reshaped_complete = complete_tensor.view(batch_size, complete_blocks, 2, block_size, d_output)
                reshaped_complete = reshaped_complete.view(batch_size, -1, 2, d_output)
                if remainder > 0: # Reshape final incomplete block
                    remainder_tensor = tensor[:, complete_blocks * 2 * block_size:, :]
                    reshaped_remainder = remainder_tensor.view(batch_size, 2, remainder, d_output).transpose(1, 2)
                    return torch.cat([reshaped_complete, reshaped_remainder], dim=1)
                else:
                    return reshaped_complete
    else:
        raise ValueError(f"Unknown interleaving strategy: {self.interleaving_strategy}")
        
    return x
```

**Refactored Implementation with Helper Function:**
```python
def _deinterleave_stereo_tensor(self, tensor, batch_size, seq_len_interleaved, d_output):
    """
    Helper function to deinterleave a stereo tensor based on the interleaving strategy.
    """
    seq_len = seq_len_interleaved // 2
    
    if self.interleaving_strategy == 'temporal':
        return tensor.view(batch_size, seq_len, 2, d_output)
    elif self.interleaving_strategy.startswith('blocking'):
        if self.interleaving_strategy == 'blocking':
            return tensor.view(batch_size, 2, seq_len, d_output).transpose(1, 2)
        else: # variable block size
            block_size = int(self.interleaving_strategy.split('-')[-1])
            if block_size <= 0:
                raise ValueError(f"Block size must be positive, got {block_size}")
            
            block_size = min(block_size, seq_len)
            complete_blocks = seq_len // block_size
            remainder = seq_len % block_size
            
            if remainder == 0:
                reshaped_tensor = tensor.view(batch_size, complete_blocks, 2, block_size, d_output)
                return reshaped_tensor.view(batch_size, -1, 2, d_output)
            else:
                complete_tensor = tensor[:, :complete_blocks * 2 * block_size, :]
                reshaped_complete = complete_tensor.view(batch_size, complete_blocks, 2, block_size, d_output)
                reshaped_complete = reshaped_complete.view(batch_size, -1, 2, d_output)
                if remainder > 0: # Reshape final incomplete block
                    remainder_tensor = tensor[:, complete_blocks * 2 * block_size:, :]
                    reshaped_remainder = remainder_tensor.view(batch_size, 2, remainder, d_output).transpose(1, 2)
                    return torch.cat([reshaped_complete, reshaped_remainder], dim=1)
                else:
                    return reshaped_complete
    else:
        raise ValueError(f"Unknown interleaving strategy: {self.interleaving_strategy}")

# Simplified reshape_stereo_output method
def reshape_stereo_output(self, x):
    if not self.is_stereo:
        return x
        
    if isinstance(x, dict):
        reshaped_dict = {}
        for key, tensor in x.items():
            batch_size, seq_len_interleaved, d_output = tensor.shape
            reshaped_tensor = self._deinterleave_stereo_tensor(tensor, batch_size, seq_len_interleaved, d_output)
            reshaped_dict[key] = reshaped_tensor
        return reshaped_dict
    
    batch_size, seq_len_interleaved, d_output = x.shape
    return self._deinterleave_stereo_tensor(x, batch_size, seq_len_interleaved, d_output)
```

## 6. Decoder Processing (tasks.py)

### Unified Decoder Creation
```python
# In LMTask.__init__()
# Both mono and stereo use the same decoder logic
decoder_input_dim = d_output

decoder = nn.Linear(decoder_input_dim, n_tokens)
```

**Key Change**: No special decoder handling for stereo - both use identical decoder logic.

## 7. Output Generation

### Mono Output
- **Shape**: `(batch, seq_len, n_tokens)`
- **Content**: Predictions for each time step
- **Processing**: Standard autoregressive generation

### Stereo Output
- **Shape**: `(batch, seq_len * 2, n_tokens)` (interleaved) or `(batch, seq_len, 2, n_tokens)` (reshaped)
- **Content**: Interleaved predictions `[L_pred, R_pred, L_pred, R_pred, ...]`
- **Processing**: Can be de-interleaved during generation using `reshape_stereo_output()`

## Summary of Key Differences

| Aspect | Mono (`is_stereo=False`) | Stereo (`is_stereo=True`) |
|--------|-------------------------|---------------------------|
| **Input Shape** | `(batch, seq_len, 1)` | `(batch, seq_len * 2, 1)` |
| **Data Processing** | Single channel | Interleaved channels |
| **Encoder** | `EmbeddingWithSqueeze` | `EmbeddingWithSqueeze` (same) |
| **Encoder Output** | `(batch, seq_len, d_model)` | `(batch, seq_len * 2, d_model)` |
| **Model Input** | `(batch, seq_len, d_model)` | `(batch, seq_len * 2, d_model)` |
| **Sequence Length** | `seq_len` | `seq_len * 2` (interleaved) |
| **Model l_max** | `sample_len` | `sample_len * 2` |
| **d_input** | `1` | `1` (always) |
| **Decoder** | Standard | Standard (same) |
| **Output Shape** | `(batch, seq_len, n_tokens)` | `(batch, seq_len * 2, n_tokens)` |
| **Output Content** | Sequential predictions | Interleaved L/R predictions |

## Key Architectural Improvements

### 1. **Unified Architecture**
- Both mono and stereo use identical `LMTask` and `EmbeddingWithSqueeze`
- No special stereo encoders or complex branching logic
- Single code path for both mono and stereo

### 2. **Data-Level Interleaving**
- Stereo channels are interleaved at the dataset level before quantization
- No architectural changes needed - just data preprocessing
- Configurable interleaving strategies (`temporal` vs `blocking`)

### 3. **Automatic Dimension Handling**
- `EmbeddingWithSqueeze` automatically handles dimension squeezing
- `d_input` always returns `1` for both mono and stereo
- No manual dimension management required

### 4. **Backward Compatibility**
- Mono processing remains completely unchanged
- Existing mono configurations work without modification
- Easy to switch between mono and stereo via configuration

### 5. **Simplified Configuration**
- Single source of truth: `sample_len` automatically calculates `__l_max`
- No need to remember special stereo configurations
- Clean, maintainable configuration files

## Interleaving Strategies

The unified stereo implementation supports two configurable interleaving strategies through the `interleaving_strategy` parameter in dataset configurations.

### Strategy 1: Temporal Interleaving `[L, R, L, R, ...]` (Default)

**Implementation:**
```python
# In AbstractAudioDataset.__getitem__()
if self.interleaving_strategy == 'temporal':
    # Interleave channels: [L, R, L, R, ...]
    # Reshape from (L, 2) to (L*2, 1)
    seq = seq.view(-1, 1)  # (L*2, 1)
```

**Data Flow:**
```
Input:  (L, 2) -> [L0, R0, L1, R1, L2, R2, ...]
Output: (L*2, 1) -> [L0, R0, L1, R1, L2, R2, ...]
```

**Characteristics:**
- **Pattern**: `[L0, R0, L1, R1, L2, R2, ...]`
- **Sequence Length**: Doubled (`L` → `L*2`)
- **Cross-Channel**: Direct temporal coupling at each time step
- **Model Learning**: Learns `L[t] ↔ R[t]` relationships through sequence

**Pros:**
- ✅ **Preserves temporal relationships** within each channel
- ✅ **Natural cross-channel dependencies** at each time step
- ✅ **Model learns L[t] ↔ R[t] relationships** through sequence structure
- ✅ **Intuitive**: Each time step contains both L and R information
- ✅ **Stereo-aware**: Model naturally learns stereo-specific patterns

**Cons:**
- ❌ **Doubles sequence length** (memory and computation overhead)
- ❌ **May confuse temporal modeling** (alternating L/R pattern)
- ❌ **Complex de-interleaving** during generation

**Use Cases:**
- **Stereo music generation** where L/R channels are highly correlated
- **Spatial audio** where cross-channel relationships are important
- **Applications requiring** strong L/R coupling

### Strategy 2: Channel Blocking `[L, L, L, ..., R, R, R, ...]`

**Implementation:**
```python
# In AbstractAudioDataset.__getitem__()
elif self.interleaving_strategy.startswith('blocking'):
    if self.interleaving_strategy == 'blocking':
        # Block channels: [L, L, L, ..., R, R, R, ...]
        # Concatenate channels: (L, 2) -> (L*2, 1)
        seq = torch.cat([seq[:, 0], seq[:, 1]], dim=0).unsqueeze(-1)  # (L*2, 1)
    else: # some block size parameter is provided
        block_size = int(self.interleaving_strategy.split('-')[-1])
        if block_size <= 0: # Validate block size
            raise ValueError(f"Block size must be positive, got {block_size}")
        seq_len = seq.shape[0]
        block_size = min(block_size, seq_len) # Truncate block size if it's larger than sequence length
        interleaved_blocks = []
        for start in range(0, seq_len, block_size):
            end = min(start + block_size, seq_len)
            interleaved_blocks.append(seq[start:end].flatten())
        seq = torch.cat(interleaved_blocks, dim=0).unsqueeze(-1)  # (L*2, 1)
```

**Data Flow:**
```
Input:  (L, 2) -> [L0, L1, L2, ..., R0, R1, R2, ...]
Output: (L*2, 1) -> [L0, L1, L2, ..., R0, R1, R2, ...]

For blocking-<size> (e.g., blocking-500):
Input:  (1028, 2) -> [L0...L1027, R0...R1027]
Output: (2056, 1) -> [L0...L499, R0...R499, L500...L999, R500...R999, L1000...L1027, R1000...R1027]
```

**Characteristics:**
- **Pattern**: `[L0, L1, L2, ..., R0, R1, R2, ...]`
- **Sequence Length**: Doubled (`L` → `L*2`)
- **Channel Separation**: L and R channels processed in separate blocks
- **Model Learning**: Learns temporal patterns within each channel independently

**Pros:**
- ✅ **Preserves channel-specific temporal patterns** (L and R processed separately)
- ✅ **Easier de-interleaving** during generation (simple reshape)
- ✅ **Clear channel boundaries** (first half = L, second half = R)
- ✅ **Reduced cross-channel interference** during training
- ✅ **Simpler generation logic** (predict L block, then R block)

**Cons:**
- ❌ **Loses cross-channel temporal relationships** (L[t] and R[t] are far apart)
- ❌ **May not learn stereo-specific patterns** (treats as two separate mono channels)
- ❌ **Reduced spatial awareness** (no direct L/R coupling)
- ❌ **Doubles sequence length** (memory and computation overhead)

**Use Cases:**
- **Independent channel processing** where L/R are less correlated
- **Multi-channel audio** where channels are processed separately
- **Applications requiring** simple generation logic
- **When cross-channel relationships** are less important

### Configuration Examples

**Temporal Interleaving (Default):**
```yaml
_name_: qautoaudio
path: musdb18stereo
bits: 16
sample_len: 12288
is_stereo: true
interleaving_strategy: temporal  # [L, R, L, R, ...]
__l_max: ${eval:${.sample_len} * 2}
```

**Channel Blocking:**
```yaml
_name_: qautoaudio
path: musdb18stereo
bits: 16
sample_len: 12288
is_stereo: true
interleaving_strategy: blocking  # [L, L, L, ..., R, R, R, ...]
__l_max: ${eval:${.sample_len} * 2}
```

### De-Interleaving During Generation

**Temporal Strategy De-Interleaving:**
```python
def deinterleave_temporal(generated, seq_len):
    """
    De-interleave temporal interleaved output.
    Input: (seq_len*2, n_tokens) - [L0, R0, L1, R1, ...]
    Output: (seq_len, 2, n_tokens) - [L0, L1, L2, ...], [R0, R1, R2, ...]
    """
    generated = generated.view(seq_len, 2, -1)  # (seq_len, 2, n_tokens)
    return generated
```

**Blocking Strategy De-Interleaving:**
```python
def deinterleave_blocking(generated, seq_len):
    """
    De-interleave blocking interleaved output.
    Input: (seq_len*2, n_tokens) - [L0, L1, L2, ..., R0, R1, R2, ...]
    Output: (seq_len, 2, n_tokens) - [L0, L1, L2, ...], [R0, R1, R2, ...]
    """
    L_block = generated[:seq_len]  # First half
    R_block = generated[seq_len:]   # Second half
    return torch.stack([L_block, R_block], dim=1)  # (seq_len, 2, n_tokens)
```

### Performance Comparison

| Aspect | Temporal Interleaving | Channel Blocking |
|--------|----------------------|------------------|
| **Memory Usage** | Same (both double sequence length) | Same (both double sequence length) |
| **Training Speed** | Same (same sequence length) | Same (same sequence length) |
| **Cross-Channel Learning** | ✅ Strong (direct L/R coupling) | ❌ Weak (separate blocks) |
| **Temporal Consistency** | ❌ May be confused (alternating pattern) | ✅ Strong (continuous blocks) |
| **Generation Complexity** | ❌ Complex (requires de-interleaving) | ✅ Simple (predict L, then R) |
| **Stereo Quality** | ✅ High (learns stereo relationships) | ❌ Lower (treats as separate channels) |
| **Use Case** | Stereo music, spatial audio | Independent channels, simple generation |

### Recommendation

**Use Temporal Interleaving when:**
- Generating stereo music where L/R channels are highly correlated
- Cross-channel relationships are important for quality
- You want the model to learn stereo-specific patterns
- Audio quality is the primary concern

**Use Channel Blocking when:**
- L/R channels are relatively independent
- You need simpler generation logic
- Cross-channel relationships are less important
- You want to treat stereo as two separate mono channels

**Default Choice:** Temporal interleaving is recommended as the default because it better captures the stereo nature of the audio and learns cross-channel relationships that are important for high-quality stereo generation.


## Executive Summary: Unified Stereo Implementation

The **unified stereo implementation** treats stereo as a **pure data preprocessing problem** with configurable interleaving strategies:

1. **Data-Level Processing**: Stereo channels are interleaved at the dataset level before quantization
2. **Unified Architecture**: Both mono and stereo use identical `LMTask` and `EmbeddingWithSqueeze`
3. **Automatic Handling**: Dimension squeezing and sequence length calculation are handled automatically
4. **Configurable Strategies**: Two interleaving strategies (`temporal` vs `blocking`) for different use cases
5. **Backward Compatibility**: Mono processing remains completely unchanged

**Key Benefits:**
- ✅ **Architectural Consistency**: Both mono and stereo use the same model architecture
- ✅ **Simplified Code**: No special stereo encoders or complex branching logic
- ✅ **Natural Cross-Channel Learning**: Interleaving allows model to learn L/R relationships
- ✅ **Backward Compatibility**: Mono processing remains completely unchanged
- ✅ **Easy Debugging**: Single code path for both mono and stereo

**Trade-offs:**
- ❌ **Sequence Length**: Stereo doubles sequence length, requiring more memory
- ❌ **Temporal Modeling**: Interleaving may affect temporal dependency learning
- ❌ **Cross-Channel Relationships**: Model must learn to relate L[t] and R[t] through sequence

This approach maintains the simplicity and consistency while enabling flexible stereo audio generation through the existing mono architecture.
