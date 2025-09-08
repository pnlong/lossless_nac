# SaShiMi Stereo Implementation v2: Unified Data-Level Approach

## Problem Analysis

The current stereo implementation diverges from mono in a critical way:

- **Mono**: Model receives raw quantized integers `(batch, seq_len, 1)` → LMTask creates embedding → `(batch, seq_len, d_model)`
- **Stereo**: Model receives raw quantized integers `(batch, seq_len, 2)` → BaseTask creates StereoEncoder → `(batch, seq_len * 2, d_model)`

**The Issue**: Stereo uses a special `StereoEncoder` that does interleaving + embedding, while mono uses the standard LMTask embedding. This creates architectural inconsistency.

## Solution: Pure Data-Level Interleaving

Make stereo a **pure data problem** where:
1. **Both mono and stereo** use the same LMTask embedding strategy
2. **Only difference**: Stereo data is interleaved at the data level to create `2x` sequence length
3. **No architectural changes** - just data preprocessing

## Implementation Strategy

### Core Principle
- **Mono**: `(batch, seq_len, 1)` → LMTask embedding → `(batch, seq_len, d_model)`
- **Stereo**: `(batch, seq_len, 2)` → **Data interleaving** → `(batch, seq_len * 2, 1)` → LMTask embedding → `(batch, seq_len * 2, d_model)`

### Key Changes Required

1. **Remove StereoEncoder**: Delete the special stereo encoder from BaseTask
2. **Add Data-Level Interleaving**: Interleave stereo channels in the dataset before quantization
3. **Update LMTask**: Handle the doubled sequence length for stereo
4. **Maintain Backwards Compatibility**: Mono processing remains unchanged

## Detailed Implementation Plan

### Phase 1: Remove StereoEncoder Architecture

#### Files to Modify: `/sashimi/s4/src/tasks/tasks.py`

**Remove the entire `_setup_stereo_embedding()` method and StereoEncoder class:**

```python
# DELETE THIS ENTIRE METHOD:
def _setup_stereo_embedding(self):
    """Setup stereo embedding for audio datasets if explicitly enabled."""
    # ... entire method to be deleted

# DELETE THIS ENTIRE CLASS:
class StereoEncoder(nn.Module):
    # ... entire class to be deleted
```

**Result**: BaseTask no longer creates any special stereo encoders. Both mono and stereo will use the standard LMTask embedding.

### Phase 2: Add Data-Level Interleaving

#### Files to Modify: `/sashimi/s4/src/dataloaders/audio.py`

**Modify `AbstractAudioDataset.__getitem__()` method:**

```python
def __getitem__(self, index):
    # ... existing loading code ...
    
    # Average non-mono signals across channels (only for mono processing)
    if seq.shape[0] > 1 and not self.is_stereo:
        seq = seq.mean(dim=0, keepdim=True)
    
    # Transpose the signal to get (L, C) where C is channels (1 for mono, 2 for stereo)
    seq = seq.transpose(0, 1)
    
    # NEW: Interleave stereo channels at data level
    if self.is_stereo and seq.shape[1] == 2:
        # Interleave channels: [L, R, L, R, ...]
        # Reshape from (L, 2) to (L*2, 1)
        seq = seq.view(-1, 1)  # (L*2, 1)
    
    # Reshape to (1, L, 1) for mono or (1, L*2, 1) for stereo
    seq = seq.unsqueeze(0)
    
    # Quantized signal
    qseq = self.quantizer(seq, self.bits)
    
    # Squeeze back to (L, 1) for mono or (L*2, 1) for stereo
    qseq = qseq.squeeze(0)
    
    # ... rest of method unchanged ...
```

**Key Changes:**
- **Mono**: `(L, 1)` → quantization → `(L, 1)` (unchanged)
- **Stereo**: `(L, 2)` → interleave → `(L*2, 1)` → quantization → `(L*2, 1)`

### Phase 3: Update Dataset Properties

#### Files to Modify: `/sashimi/s4/src/dataloaders/audio.py`

**Update `QuantizedAutoregressiveAudio` class:**

```python
@property
def d_input(self):
    # Always return 1 since we interleave stereo to single channel
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

### Phase 4: Update Model Configuration

#### Files to Modify: `/sashimi/s4/train.py`

**Update sequence length and stereo configuration:**

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

### Phase 5: Add Stereo Output Reshaping

#### Files to Modify: `/sashimi/s4/src/models/sequence/backbones/sashimi.py`

**Add stereo configuration parameters:**

```python
def __init__(
    self,
    # ... existing parameters ...
    is_stereo=False,           # New: whether processing stereo audio
    interleaving_strategy='temporal',  # New: interleaving strategy for stereo
):
    # ... existing initialization ...
    
    # Stereo configuration
    self.is_stereo = is_stereo
    self.interleaving_strategy = interleaving_strategy
```

**Add stereo output reshaping method:**

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
    elif self.interleaving_strategy == 'blocking':
        # Blocking interleaving: [L, L, L, ..., R, R, R, ...] -> [L, L, L, ...], [R, R, R, ...]
        x = x.view(batch_size, 2, seq_len, d_output).transpose(1, 2)
    else:
        raise ValueError(f"Unknown interleaving strategy: {self.interleaving_strategy}")
        
    return x
```

**Update forward method to apply stereo reshaping:**

```python
def forward(self, x, state=None, **kwargs):
    # ... existing forward logic ...
    
    # Apply output head
    x = self.output_head(x)
    
    # Reshape stereo output if needed
    if self.is_stereo:
        x = self.reshape_stereo_output(x)
    
    return x, None
```

### Phase 6: Update Forward Pass

#### Files to Modify: `/sashimi/s4/train.py`

**Simplify forward pass logic:**

```python
def forward(self, batch):
    x, y, *z = batch
    
    # No special stereo handling needed - data is already interleaved
    # x shape: (batch, seq_len, 1) for mono or (batch, seq_len*2, 1) for stereo
    
    x, w = self.encoder(x, **z)  # Standard LMTask embedding
    x, state = self.model(x, **w, state=self._state)  # Model processes interleaved sequence
    x, w = self.decoder(x, state=state, **z)
    return x, y, w
```

**Key Change**: No special stereo handling in forward pass - the interleaving is handled at data level and output reshaping is handled by the model.

### Phase 6: Update Generation Logic

#### Files to Modify: `/sashimi/s4/generate.py`

**Add de-interleaving for stereo generation:**

```python
def generate_stereo(self, length, **kwargs):
    # Generate interleaved sequence
    generated = self.generate(length * 2, **kwargs)  # Double length for stereo
    
    # De-interleave: [L, R, L, R, ...] -> [L, L, L, ...], [R, R, R, ...]
    if generated.dim() == 2:  # (seq_len*2, 1)
        generated = generated.squeeze(-1)  # (seq_len*2,)
    
    # Reshape to separate channels
    generated = generated.view(-1, 2)  # (seq_len, 2)
    
    return generated
```

## Configurable Interleaving Strategies

The implementation supports two interleaving strategies through the `interleaving_strategy` parameter in dataset configurations.

### Strategy 1: Temporal Interleaving `[L, R, L, R, ...]` (Default)
```python
# Input: (L, 2) -> Output: (L*2, 1)
# [L0, R0, L1, R1, L2, R2, ...]
seq = seq.view(-1, 1)
```

**Pros:**
- Preserves temporal relationships within each channel
- Natural cross-channel dependencies at each time step
- Model learns L[t] ↔ R[t] relationships

**Cons:**
- Doubles sequence length
- May confuse temporal modeling

### Strategy 2: Channel Blocking `[L, L, L, ..., R, R, R, ...]`
```python
# Input: (L, 2) -> Output: (L*2, 1)  
# [L0, L1, L2, ..., R0, R1, R2, ...]
seq = torch.cat([seq[:, 0], seq[:, 1]], dim=0).unsqueeze(-1)
```

**Pros:**
- Preserves channel-specific temporal patterns
- Easier to de-interleave during generation

**Cons:**
- Loses cross-channel temporal relationships
- May not learn stereo-specific patterns

### Configuration Example
```yaml
_name_: qautoaudio
path: musdb18stereo
bits: 16
sample_len: 12288
is_stereo: true
interleaving_strategy: temporal  # Options: temporal, blocking
__l_max: ${eval:${.sample_len} * 2}  # Double the sample_len for stereo
```

**Important**: The `__l_max` parameter must be doubled for stereo configurations to account for the interleaved sequence length. Using `${eval:${.sample_len} * 2}` ensures the model is initialized with the correct maximum sequence length and automatically adapts if `sample_len` is changed.

## Implementation Steps

### Step 1: Remove StereoEncoder
1. Delete `_setup_stereo_embedding()` method from BaseTask
2. Delete StereoEncoder class
3. Remove stereo-specific encoder creation logic

### Step 2: Add Data-Level Interleaving
1. Modify `AbstractAudioDataset.__getitem__()` to interleave stereo channels
2. Update tensor reshaping to handle interleaved data
3. Ensure quantization works with interleaved data

### Step 3: Update Dataset Properties
1. Set `d_input` to always return `1`
2. Update sequence length calculations for stereo
3. Ensure backwards compatibility with mono

### Step 4: Update Model Configuration
1. Modify `l_max` calculation for stereo (double the length)
2. Ensure `d_input` is always `1`
3. Update any stereo-specific model parameters

### Step 5: Update Generation
1. Add de-interleaving logic for stereo generation
2. Ensure generated audio maintains stereo structure
3. Test generation quality

### Step 6: Testing
1. Test backwards compatibility with mono configurations
2. Test stereo data loading and interleaving
3. Test model training with interleaved stereo data
4. Test stereo generation and de-interleaving

## Expected Benefits

1. **Architectural Consistency**: Both mono and stereo use the same LMTask embedding
2. **Simplified Code**: No special stereo encoders or complex branching logic
3. **Natural Cross-Channel Learning**: Interleaving allows model to learn L/R relationships
4. **Backwards Compatibility**: Mono processing remains completely unchanged
5. **Easier Debugging**: Single code path for both mono and stereo

## Potential Challenges

1. **Sequence Length**: Stereo doubles sequence length, requiring more memory
2. **Temporal Modeling**: Interleaving may affect temporal dependency learning
3. **Generation Quality**: De-interleaving must preserve stereo quality
4. **Cross-Channel Relationships**: Model must learn to relate L[t] and R[t] through sequence

## Summary

This approach treats stereo as a **pure data preprocessing problem** with configurable interleaving strategies:
- Stereo channels are processed at the data level using configurable strategies
- Both mono and stereo use identical LMTask embedding and model architecture
- Both strategies double the sequence length: `L` → `L*2`
- No architectural changes needed - just data preprocessing and generation post-processing

## Implementation Status

### ✅ Completed Changes

1. **Removed StereoEncoder**: Deleted special stereo encoder from BaseTask
2. **Added Configurable Interleaving**: Added `interleaving_strategy` parameter with 2 options:
   - `temporal`: `[L, R, L, R, ...]` (default)
   - `blocking`: `[L, L, L, ..., R, R, R, ...]`
3. **Updated Data Loading**: Modified `AbstractAudioDataset.__getitem__()` to support both strategies
4. **Updated Dataset Properties**: `d_input` always returns `1` (stereo is interleaved to single channel)
5. **Updated Model Configuration**: `l_max` calculation doubles for both stereo strategies
6. **Updated Forward Pass**: Simplified to handle interleaved data uniformly
7. **Created Configuration Files**: Added dataset configs for both interleaving strategies
8. **Created Test Suite**: Comprehensive tests for both strategies
9. **Fixed Model Initialization**: Updated dataset config to set correct `__l_max` for stereo (doubled sequence length)

### 🔧 Key Files Modified

- `/sashimi/s4/src/tasks/tasks.py`: Removed StereoEncoder, simplified BaseTask
- `/sashimi/s4/src/dataloaders/audio.py`: Added interleaving strategies, updated dataset properties
- `/sashimi/s4/train.py`: Updated model configuration and forward pass logic
- `/sashimi/s4/configs/dataset/musdb18stereo.yaml`: Added interleaving_strategy parameter
- `/sashimi/test_unified_stereo.py`: Comprehensive test suite

### 🎯 Benefits Achieved

1. **Architectural Consistency**: Both mono and stereo use the same LMTask embedding
2. **Configurable Processing**: Two interleaving strategies for different use cases
3. **Backwards Compatibility**: Mono processing remains completely unchanged
4. **Simplified Code**: Single code path for both mono and stereo
5. **Easy Experimentation**: Switch strategies via configuration without code changes

This maintains the simplicity and consistency your advisor requested while enabling flexible stereo audio generation through the existing mono architecture.
