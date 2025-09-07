# Sashimi Stereo vs Mono: Input Transformation Analysis

This document traces the complete data flow from raw audio input to final model output, comparing mono and stereo processing paths in the Sashimi audio generation model.

## Overview

The Sashimi model processes audio through several key stages:
1. **Data Loading & Preprocessing** (audio.py)
2. **Encoder Processing** (tasks.py)
3. **Model Forward Pass** (train.py)
4. **Decoder Processing** (decoders.py)
5. **Final Output Generation**

The key difference between mono and stereo lies in how multi-channel audio is handled during encoding and how the model processes the interleaved sequence.

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

# Add batch dimension: (1, length, 2)
seq = seq.unsqueeze(0)

# Quantize and squeeze back to (length, 2)
qseq = self.quantizer(seq, self.bits).squeeze(0)
```

**Stereo Data Shape**: `(length, 2)` - Two-channel audio

### Dataset Properties
```python
# In QuantizedAutoregressiveAudio
@property
def d_input(self):
    return 2 if getattr(self, 'is_stereo', False) else 1

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

## 2. Encoder Processing (tasks.py)

### Mono Encoder Path
When `is_stereo=False`, the `BaseTask._setup_stereo_embedding()` creates an `IdentityEncoder`:

```python
class IdentityEncoder(nn.Module):
    def forward(self, x, **kwargs):
        return x, kwargs  # Pass-through with no transformation
```

**Mono Encoder Output**: `(batch, length, 1)` - No transformation applied

### Stereo Encoder Path
When `is_stereo=True`, a `StereoEncoder` is created:

```python
class StereoEncoder(nn.Module):
    def __init__(self, n_tokens, d_model):
        super().__init__()
        # Create embedding layer to convert integer indices to dense features
        self.embedding = nn.Embedding(n_tokens, d_model)
        nn.init.normal_(self.embedding.weight, mean=0, std=d_model**-.5)
    
    def forward(self, x, **kwargs):
        # x shape: (batch, length, channels)
        batch_size, seq_len, channels = x.shape
        
        # Interleave the channels to create a single sequence
        interleaved = x.view(batch_size, -1)  # (batch, length * channels)
        
        # Convert to long tensor for embedding layer
        interleaved_long = interleaved.long()
        
        # Convert integer indices to dense embeddings
        embedded = self.embedding(interleaved_long)  # (batch, length * channels, d_model)
        
        return embedded, kwargs
```

**Stereo Encoder Output**: `(batch, length * 2, d_model)` - Interleaved channels with embedding

### Key Differences in Encoder Processing

1. **Mono**: Raw quantized values passed through unchanged
2. **Stereo**: 
   - Channels are interleaved: `[L, R, L, R, L, R, ...]`
   - Each quantized value is embedded into `d_model` dimensions
   - Sequence length is doubled: `length` → `length * 2`

### Why Stereo Needs Embedding (Critical Insight)

The stereo encoder **must** apply embedding because of how the LMTask handles encoder creation:

```python
# In LMTask.__init__()
# Check if stereo encoder was already set up by BaseTask
if hasattr(self, 'encoder') and self.encoder is not None:
    # ✅ Stereo case: BaseTask already created StereoEncoder with embedding
    # Don't overwrite - just add scaling if needed
    pass
else:
    # ❌ Mono case: No encoder exists, create new one
    if d_input > 1:
        # This creates MultiChannelEmbedding (separate embeddings per channel)
        # Output: (batch, length, channels * d_model)
    else:
        # This creates single embedding
        # Output: (batch, length, d_model)
```

**The Key Issue**: If the stereo encoder output `(batch, length * 2, 1)` (raw quantized values), then:

1. **LMTask would detect no existing encoder** (`self.encoder is None`)
2. **LMTask would create a new encoder** in the `else` branch
3. **This would create MultiChannelEmbedding** which expects `(batch, length, channels)` input
4. **But we'd be passing `(batch, length * 2, 1)`** - wrong shape!

**The Solution**: The `StereoEncoder` must output `(batch, length * 2, d_model)` so that:
1. **LMTask detects existing encoder** (`self.encoder is not None`)
2. **LMTask skips encoder creation** and uses the existing `StereoEncoder`
3. **Model receives proper `d_model` dimensional input** instead of raw quantized values

This is why stereo **cannot** output `(batch, length * 2, 1)` - it would break the encoder detection logic and cause shape mismatches in the LMTask initialization.

## Task Selection: Both Mono and Stereo Use LMTask

**Yes, both mono and stereo systems use `LMTask`!** Here's how:

### Task Registry
```python
# In tasks.py
registry = {
    'base': BaseTask,
    'lm': LMTask,           # ← Used for audio generation
    'adaptivelm': AdaptiveLMTask,
    'imagenet': ImageNetTask,
    'forecasting': ForecastingTask,
    'video': VideoTask,
}
```

### Task Instantiation in train.py
```python
# In SequenceLightningModule.setup()
task_config = self.hparams.task.copy()
self.task = utils.instantiate(
    tasks.registry, task_config, dataset=self.dataset, model=self.model
)
```

### How LMTask Handles Both Cases

**LMTask** is designed to handle both mono and stereo through conditional logic:

1. **Inherits from BaseTask**: `class LMTask(BaseTask)`
2. **BaseTask sets up stereo encoder**: `self._setup_stereo_embedding()` 
3. **LMTask detects existing encoder**: 
   - **Stereo**: `self.encoder is not None` → Uses `StereoEncoder`
   - **Mono**: `self.encoder is None` → Creates new encoder

### The Unified Architecture

```python
# Both mono and stereo follow this path:
BaseTask.__init__() 
    ↓
BaseTask._setup_stereo_embedding()
    ↓ (if stereo enabled)
StereoEncoder created
    ↓
LMTask.__init__()
    ↓
LMTask detects existing encoder
    ↓
Uses StereoEncoder (stereo) OR creates new encoder (mono)
```

**Key Point**: There's only **one task class** (`LMTask`) that intelligently adapts its behavior based on whether stereo processing is enabled. The stereo/mono distinction is handled entirely within the encoder setup logic, not by using different task classes.

## Critical Discovery: Model Input Dimensions

**No, the model does NOT receive proper `d_model` dimensional input in both cases!**

### Mono Model Input
```python
# Mono processing path:
Raw Audio: Continuous float values [-1, 1]
    ↓
Quantization: linear_encode() or mu_law_encode()
    ↓
Quantized Values: Integer indices [0, 255] for 8-bit audio
    ↓
Input: (batch, seq_len, 1)           # Raw quantized integer indices
IdentityEncoder: return x, kwargs    # Pass-through unchanged  
Model receives: (batch, seq_len, 1)  # Raw quantized integer indices!
```

### Stereo Model Input  
```python
# Stereo processing path:
Raw Audio: Continuous float values [-1, 1] 
    ↓
Quantization: linear_encode() or mu_law_encode()
    ↓
Quantized Values: Integer indices [0, 255] for 8-bit audio
    ↓
Input: (batch, seq_len, 2)                    # Raw quantized integer indices
StereoEncoder: embedding(interleaved_long)     # Convert indices to dense embeddings
Model receives: (batch, seq_len * 2, d_model) # Dense embeddings!
```

## The Architectural Difference

## The Quantization Process

### Quantization Details
```python
# For 8-bit audio (default):
q_levels = 1 << 8 = 256 levels
Quantized values: [0, 1, 2, ..., 255]  # Integer indices

# Quantization functions:
def quantize(samples, bits=8, epsilon=0.01):
    q_levels = 1 << bits  # 256 for 8-bit
    samples *= q_levels - epsilon
    samples += epsilon / 2
    return samples.long()  # Convert to integer indices
```

### What the Model Actually Receives

- **Stereo**: Model receives **dense embeddings** `(batch, seq_len * 2, d_model)`
  - Each integer index [0-255] is converted to a `d_model`-dimensional vector
  - Example: index `42` → `[0.1, -0.3, 0.7, ...]` (d_model dimensions)
  
- **Mono**: Model receives **raw quantized integer indices** `(batch, seq_len, 1)`
  - Each value is just an integer [0-255] representing the quantized audio level
  - Example: `[42, 128, 200, ...]` (raw integer indices)

**Implication**: The mono model must be designed to handle raw integer indices directly, while the stereo model receives pre-embedded dense representations. This suggests the model architecture itself may need to adapt based on the input type.

## 3. Model Configuration (train.py)

### Sequence Length Adjustment for Stereo
```python
# In SequenceLightningModule.setup()
if hasattr(self.dataset, 'is_stereo') and self.dataset.is_stereo and hasattr(self.dataset, 'd_input') and self.dataset.d_input > 1:
    raw_d_input = self.dataset.d_input
    if 'l_max' in model_config and model_config['l_max'] is not None:
        original_l_max = model_config['l_max']
        stereo_l_max = original_l_max * raw_d_input
        model_config['l_max'] = stereo_l_max
```

**Key Point**: The model's `l_max` parameter is doubled for stereo to handle the interleaved sequence length.

### Input Dimension Handling
```python
# For mono, ensure d_input matches the actual input dimension
raw_d_input = getattr(self.dataset, 'd_input', 1)
model_config['d_input'] = raw_d_input
```

## 4. Forward Pass Processing (train.py)

### Mono Forward Pass
```python
def forward(self, batch):
    x, y, *z = batch
    
    # Handle mono data format: squeeze out singleton channel dimension
    if (hasattr(self.dataset, 'is_stereo') and not self.dataset.is_stereo and 
        hasattr(self.dataset, 'd_input') and self.dataset.d_input == 1 and 
        x.dim() == 3 and x.shape[-1] == 1):
        x = x.squeeze(-1)  # Remove singleton channel dimension: (batch, seq_len, 1) -> (batch, seq_len)
    
    x, w = self.encoder(x, **z)  # IdentityEncoder: no change
    x, state = self.model(x, **w, state=self._state)  # Model processes (batch, seq_len, d_model)
    x, w = self.decoder(x, state=state, **z)
    return x, y, w
```

**Mono Model Input**: `(batch, seq_len, d_model)` - Standard sequence processing

### Stereo Forward Pass
```python
def forward(self, batch):
    x, y, *z = batch
    
    # NO squeezing for stereo - preserve channel dimension
    # x shape: (batch, seq_len, 2)
    
    x, w = self.encoder(x, **z)  # StereoEncoder: interleaves and embeds
    # x shape after encoder: (batch, seq_len * 2, d_model)
    
    x, state = self.model(x, **w, state=self._state)  # Model processes interleaved sequence
    # x shape after model: (batch, seq_len * 2, d_model)
    
    x, w = self.decoder(x, state=state, **z)
    return x, y, w
```

**Stereo Model Input**: `(batch, seq_len * 2, d_model)` - Interleaved sequence processing

## 5. Decoder Processing (tasks.py)

### LMTask Decoder Creation
```python
# In LMTask.__init__()
# For stereo, d_output might be per-channel, so adjust decoder accordingly
decoder_input_dim = d_output

if d_input > 1:
    # For stereo, model outputs predictions for each channel
    # Decoder should output per-channel predictions
    decoder_input_dim = d_output // d_input if d_output > d_input else d_output

decoder = nn.Linear(decoder_input_dim, n_tokens)
```

**Key Point**: For stereo, the decoder input dimension is adjusted to handle per-channel predictions.

### Weight Tying
```python
if tied and d_input == 1:
    assert d_model == d_output
    decoder.weight = self.encoder[0].weight  # Only for mono case
```

**Weight Tying**: Only enabled for mono case (`d_input == 1`), not for stereo.

## 6. Output Generation

### Mono Output
- **Shape**: `(batch, seq_len, n_tokens)`
- **Content**: Predictions for each time step
- **Processing**: Standard autoregressive generation

### Stereo Output
- **Shape**: `(batch, seq_len * 2, n_tokens)`
- **Content**: Interleaved predictions `[L_pred, R_pred, L_pred, R_pred, ...]`
- **Processing**: Must be de-interleaved during generation

## Summary of Key Differences

| Aspect | Mono (`is_stereo=False`) | Stereo (`is_stereo=True`) |
|--------|-------------------------|---------------------------|
| **Input Shape** | `(batch, seq_len, 1)` | `(batch, seq_len, 2)` |
| **Encoder** | `IdentityEncoder` (pass-through) | `StereoEncoder` (interleaves + embeds) |
| **Encoder Output** | `(batch, seq_len, 1)` | `(batch, seq_len * 2, d_model)` |
| **Model Input** | `(batch, seq_len, d_model)` | `(batch, seq_len * 2, d_model)` |
| **Sequence Length** | `seq_len` | `seq_len * 2` (interleaved) |
| **Model l_max** | `original_l_max` | `original_l_max * 2` |
| **Decoder Input Dim** | `d_output` | `d_output // 2` (per-channel) |
| **Weight Tying** | Enabled | Disabled |
| **Output Shape** | `(batch, seq_len, n_tokens)` | `(batch, seq_len * 2, n_tokens)` |
| **Output Content** | Sequential predictions | Interleaved L/R predictions |

## Critical Design Decisions

1. **Interleaving Strategy**: Stereo channels are interleaved rather than processed separately, allowing the model to learn cross-channel dependencies.

2. **Sequence Length Doubling**: The model must handle twice the sequence length for stereo, requiring careful `l_max` configuration.

3. **Embedding Integration**: Stereo processing includes embedding layers that convert quantized values to dense representations, while mono uses raw quantized values.

4. **Decoder Adaptation**: The decoder is adjusted to handle per-channel predictions for stereo output.

5. **Weight Tying Limitation**: Weight tying between encoder and decoder is disabled for stereo due to the different processing paths.

This architecture allows the model to learn both intra-channel temporal dependencies and inter-channel spatial dependencies in stereo audio, while maintaining compatibility with mono processing through the identity encoder path.

## Addressing Key Architectural Questions

### Question 1: Why Not Simple Interleaving Without Embedding?

**Your suggestion**: Interleave stereo channels into a single "mono" waveform with `2n` samples, keeping model architecture unchanged.

**Why the current approach is used instead**:

1. **Encoder Detection Logic**: The stereo encoder must output `d_model` dimensions to satisfy LMTask's encoder detection system
2. **Model Architecture Compatibility**: The model expects either:
   - Raw quantized integers `(batch, seq_len, 1)` for mono
   - Dense embeddings `(batch, seq_len, d_model)` for stereo
3. **Cross-Channel Learning**: Interleaving with embedding allows the model to learn relationships between L/R channels through the dense representations

**Alternative approach you suggested**:
```python
# Your suggested approach:
Stereo: (batch, seq_len, 2) → Interleave → (batch, seq_len * 2, 1)
# Model receives: (batch, seq_len * 2, 1) - raw integers
# This would require the model to handle raw integers for stereo
```

**Problem**: This would require the model to handle raw quantized integers for stereo, but the current architecture expects dense embeddings for stereo processing.

### Question 2: Why Not `(batch, seq_len, d_model)` Instead of `(batch, seq_len * 2, d_model)`?

**Your suggestion**: Use dense embeddings that capture stereo nature, outputting `(batch, seq_len, d_model)`.

**Why interleaving is used**:

1. **Temporal Modeling**: Interleaving `[L, R, L, R, ...]` allows the model to learn:
   - Temporal dependencies within each channel
   - Cross-channel dependencies at each time step
   - Sequential relationships between L/R samples

2. **Sequence Model Compatibility**: Sashimi is a sequence model that processes temporal sequences. Interleaving creates a single temporal sequence that captures both channels.

3. **Cross-Channel Dependencies**: The model can learn that `L[t]` and `R[t]` are related through the interleaved sequence structure.

**Alternative approach you suggested**:
```python
# Your suggested approach:
Stereo: (batch, seq_len, 2) → Embed → (batch, seq_len, d_model)
# Model receives: (batch, seq_len, d_model) - stereo-aware embeddings
```

**Problem**: This would require the model to understand that each `d_model` vector represents both L and R channels simultaneously, which is more complex than the current interleaved approach.

### Why the Current Approach is Better

1. **Simplicity**: Interleaving creates a single sequence that the existing model architecture can process
2. **Cross-Channel Learning**: The model naturally learns L/R relationships through the interleaved sequence
3. **Temporal Consistency**: Each time step contains information from both channels
4. **Architecture Compatibility**: Works with existing sequence model architectures without major modifications

### The Trade-offs

**Current Approach**:
- ✅ Simple interleaving strategy
- ✅ Cross-channel learning through sequence
- ✅ Compatible with existing architecture
- ❌ Doubles sequence length
- ❌ Requires embedding for stereo

**Your Suggested Approach**:
- ✅ Maintains original sequence length
- ✅ Could avoid embedding requirement
- ❌ More complex stereo-aware embeddings
- ❌ Requires model to understand stereo structure
- ❌ May lose temporal cross-channel dependencies

## Executive Summary: Stereo Implementation Overview

The Sashimi stereo implementation uses an **interleaving strategy with embedding** to process stereo audio through the existing sequence model architecture. Here's how it works end-to-end: Raw stereo audio `(batch, seq_len, 2)` is quantized to integer indices `[0-(2^8 - 1)]` (or `[0-(2^16 - 1)]` if 16-bit is specified instead) using linear/mu-law quantization, then the `StereoEncoder` interleaves the channels `[L, R, L, R, ...]` and applies embedding to convert each integer to a `d_model`-dimensional vector, resulting in `(batch, seq_len * 2, d_model)`. The model processes this interleaved sequence, learning both temporal dependencies within each channel and cross-channel relationships between L/R samples. The decoder outputs predictions for each interleaved sample, which are then de-interleaved during generation. This approach maintains compatibility with the existing mono architecture while enabling the model to learn stereo-specific patterns through the temporal sequence structure. The key trade-off is doubling the sequence length in exchange for natural cross-channel learning and architectural simplicity.
