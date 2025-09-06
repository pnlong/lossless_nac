# SaShiMi Stereo Implementation Plan

## Current Mono Setup Analysis

### 1. Input Tensor Shape
- **Current**: `(batch, length, 1)` - mono audio with single channel dimension
- **Model expectation**: `d_input = 1` in QuantizedAutoregressiveAudio class
- **Data processing**: Explicitly converts multi-channel to mono by averaging

### 2. Data Loader Behavior
- **Location**: `/sashimi/s4/src/dataloaders/audio.py`, lines 142-144
- **Current behavior**: `seq = seq.mean(dim=0, keepdim=True)` converts stereo to mono
- **File format**: MUSDB18 originally stored as `(n_samples, 2)` stereo arrays
- **Conversion**: `musdb18mono_wav.py` splits stereo into separate mono files

### 3. Model Architecture
- **Backbone**: SaShiMi processes sequences through U-Net style architecture
- **Input processing**: `forward(self, x)` expects `(batch, length, d_input)`
- **Output heads**: Support both categorical and DML outputs
- **Current limitation**: Hardcoded for `d_input = 1`

### 4. Output Dimensions
- **Categorical**: `n_classes = 1 << bits` (e.g., 256 for 8-bit)
- **DML**: `3 * n_mixtures` parameters per timestep
- **No stereo-specific changes needed** in output heads

## Stereo Implementation Strategy

### Joint Channel Processing (Primary Approach)
- **Goal**: Capture cross-channel relationships by processing both channels simultaneously
- Modify input to accept `(batch, length, 2)` for stereo
- Update model to process both channels together through the U-Net architecture
- **Benefits**: Enables cross-channel modeling, more efficient than independent processing
- **Requirements**: Architectural modifications to handle increased channel dimension

### Backwards Compatibility
- Add optional `is_stereo` parameter to dataset YAML configurations
- **Default**: `is_stereo: false` to maintain existing mono behavior
- **When enabled**: `is_stereo: true` activates stereo processing mode
- Ensures existing mono configurations continue to work without changes

### Parameter Integration
- `is_stereo` parameter flows from YAML config to dataset class
- Dataset class passes stereo flag to data loading logic
- Model's `d_input` property dynamically returns 1 (mono) or 2 (stereo)
- Output heads automatically scale to match input dimensions
- Training and evaluation scripts remain agnostic to mono/stereo mode

## Implementation Steps

### Phase 1: Data Loading Changes
1. Add `is_stereo` parameter to dataset YAML configurations (defaults to `false`)
2. Modify `AbstractAudioDataset.__getitem__()` to conditionally preserve stereo channels
3. Update channel averaging logic to respect `is_stereo` flag
4. Update tensor reshaping: `(1, L, 1)` for mono, `(1, L, 2)` for stereo
5. Create `musdb18stereo.yaml` configuration with `is_stereo: true`

### Phase 2: Model Architecture Updates
1. Update `d_input` property to return 2 when `is_stereo: true`, 1 otherwise
2. Modify SaShiMi backbone to handle variable input dimensions
3. Ensure all sequence layers can handle increased channel dimension
4. Update forward pass to maintain stereo dimensions throughout U-Net
5. Verify cross-channel relationships are captured in the joint processing

### Phase 3: Output Head Considerations
1. Verify output heads work with stereo inputs
2. Categorical head: `(batch, length, 2, n_classes)` output
3. DML head: `(batch, length, 2, 3*n_mixtures)` output
4. Update loss functions to handle multi-channel predictions

### Phase 4: Configuration and Training
1. Create `musdb18stereo.yaml` configuration with `is_stereo: true`
2. Ensure existing mono configurations remain unchanged (backwards compatibility)
3. Update training scripts to handle variable input dimensions
4. Implement stereo-specific evaluation metrics
5. Test end-to-end stereo training pipeline

## Key Technical Considerations

### Memory and Compute Scaling
- Stereo doubles input feature dimensions
- Need to monitor GPU memory usage
- May require batch size adjustments

### Cross-Channel Dependencies
- Current model treats channels independently
- Stereo implementation should model inter-channel relationships
- Consider channel-wise attention or convolution operations

### Dataset Size and Training
- Stereo files provide 2x more training data
- Need to ensure proper train/val/test splits
- Consider joint vs. independent channel training objectives

---

## Detailed Implementation Plan

### Phase 1: Dataset Configuration Changes

#### New Files to Create:
1. **`/sashimi/s4/configs/dataset/musdb18stereo.yaml`**
   ```yaml
   _name_: qautoaudio
   path: musdb18stereo
   bits: 16
   sample_len: 12288
   train_percentage: 0.88
   quantization: linear
   drop_last: true
   context_len: null
   pad_len: null
   is_stereo: true  # New parameter
   __l_max: ${.sample_len}
   ```

2. **`/sashimi/s4/configs/experiment/audio/sashimi-musdb18stereo.yaml`**
   ```yaml
   # @package _global_
   defaults:
     - /trainer: default
     - /loader: default
     - /dataset: musdb18stereo  # Reference new stereo dataset
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
     train_chunk_size: 2000
     val_chunk_size: 200

   task:
     metrics:
       - bpb
       - accuracy
       - accuracy@3
       - accuracy@5
       - accuracy@10

   encoder: embedding

   decoder:
     _name_: sequence
     mode: last

   loader:
     batch_size: 1

   trainer:
     max_epochs: 1000
     limit_train_batches: 2000
     limit_val_batches: 200

   optimizer:
     lr: 0.004
   ```

3. **`/sashimi/musdb18stereo_wav.py`** (New preprocessing script)
   - Copy from `musdb18mono_wav.py`
   - Remove channel splitting logic (keep stereo as-is)
   - Change output directory to `musdb18stereo`
   - Modify filename generation to not include channel indices

### Phase 2: Core Data Loading Modifications

#### `/sashimi/s4/src/dataloaders/audio.py` Changes:

1. **Add `is_stereo` parameter to `AbstractAudioDataset.__init__()`**
   ```python
   def __init__(
       self,
       # ... existing parameters ...
       is_stereo=False,  # New parameter with default False
       **kwargs,
   ):
   ```

2. **Modify channel averaging logic in `__getitem__()`**
   ```python
   # Conditionally average non-mono signals across channels
   if seq.shape[0] > 1 and not self.is_stereo:
       seq = seq.mean(dim=0, keepdim=True)
   ```

3. **Update tensor reshaping logic**
   ```python
   # Transpose the signal to get (L, C) where C=1 for mono, C=2 for stereo
   seq = seq.transpose(0, 1)

   # Reshape to (1, L, C)
   seq = seq.unsqueeze(0)
   ```

4. **Update `QuantizedAutoregressiveAudio.init_defaults()`**
   ```python
   def init_defaults(self):
       return {
           # ... existing parameters ...
           'is_stereo': False,  # New parameter
       }
   ```

5. **Modify `QuantizedAutoregressiveAudio.d_input` property**
   ```python
   @property
   def d_input(self):
       return 2 if getattr(self, 'is_stereo', False) else 1
   ```

6. **Pass `is_stereo` to dataset instances in `setup()`**
   ```python
   self.dataset_train = QuantizedAudioDataset(
       path=self.data_dir,
       bits=self.bits,
       # ... other parameters ...
       is_stereo=self.is_stereo,  # Pass stereo flag
   )
   ```

### Phase 3: Model Architecture Updates

#### `/sashimi/s4/src/models/sequence/backbones/sashimi.py` Changes:

1. **No changes needed to core Sashimi architecture** - the model already handles variable input dimensions through the `d_model` parameter

2. **Verify `d_output` property handles stereo correctly**
   ```python
   @property
   def d_output(self):
       """Output dimension depends on output head type"""
       if self.output_head_type == "categorical":
           return self.n_classes  # This will be per-channel for stereo
       elif self.output_head_type == "dml":
           return 3 * self.n_mixtures  # This will be per-channel for stereo
       else:
           raise ValueError(f"Unknown output head type: {self.output_head_type}")
   ```

### Phase 4: Task and Encoder Updates

#### `/sashimi/s4/src/tasks/tasks.py` Changes:

1. **Update ReversibleInstanceNorm1dInput instantiation**
   ```python
   if norm == 'revnorm':
       self.encoder = ReversibleInstanceNorm1dInput(self.dataset.d_input, transposed=False)
   ```
   - This will automatically use `d_input = 2` for stereo datasets

### Phase 5: Training and Generation Scripts

#### `/sashimi/s4/train.py` Changes:
- **No changes needed** - the dynamic `d_input` property will automatically handle stereo

#### `/sashimi/s4/generate.py` Changes:
- **No changes needed** - the dynamic `d_input` property will automatically handle stereo

### Phase 6: Output Processing Updates

#### Loss Functions and Metrics:
1. **Verify categorical loss handles multi-channel outputs**
   - CrossEntropyLoss will automatically handle `(batch, length, channels, n_classes)` → `(batch, length, channels)`

2. **Verify DML loss handles multi-channel outputs**
   - DML loss function needs to handle `(batch, length, channels, 3*n_mixtures)`

3. **Update evaluation metrics** for stereo (if needed)
   - May need to average metrics across channels or report per-channel metrics

## Implementation Order

1. **Create configuration files** (musdb18stereo.yaml, sashimi-musdb18stereo.yaml)
2. **Create preprocessing script** (musdb18stereo_wav.py)
3. **Modify AbstractAudioDataset** to add `is_stereo` parameter and conditional channel processing
4. **Update QuantizedAutoregressiveAudio** to handle dynamic `d_input`
5. **Test backwards compatibility** with existing mono configurations
6. **Test stereo preprocessing** and data loading
7. **Verify model training** with stereo inputs
8. **Update evaluation metrics** if needed

## Testing Strategy

1. **Backwards Compatibility Test**: Verify existing mono configs still work
2. **Stereo Data Loading Test**: Verify stereo files load with shape `(batch, length, 2)`
3. **Model Forward Pass Test**: Verify model handles stereo inputs correctly
4. **Training Loop Test**: Verify training completes without errors
5. **Generation Test**: Verify generation works with stereo models
6. **Cross-Channel Validation**: Verify stereo models learn cross-channel relationships

---

## Implementation Progress Tracking

### Step 1: Create Stereo Dataset Configuration (musdb18stereo.yaml)
- **Status**: ✅ Completed
- **Files Modified**: `/sashimi/s4/configs/dataset/musdb18stereo.yaml` (new file)
- **Changes**:
  - Copied from `musdb18mono.yaml`
  - Added `is_stereo: true` parameter
  - Changed path to `musdb18stereo`
- **Purpose**: Enable stereo dataset configuration with backwards compatibility

### Step 2: Create Stereo Experiment Configuration (sashimi-musdb18stereo.yaml)
- **Status**: ✅ Completed
- **Files Modified**: `/sashimi/s4/configs/experiment/audio/sashimi-musdb18stereo.yaml` (new file)
- **Changes**:
  - Copied from `sashimi-musdb18mono.yaml`
  - Updated dataset reference to `musdb18stereo`
  - Inherits all stereo settings from dataset config
- **Purpose**: Enable stereo training experiments with proper configuration

### Step 3: Add is_stereo Parameter to AbstractAudioDataset
- **Status**: ✅ Completed
- **Files Modified**: `/sashimi/s4/src/dataloaders/audio.py`
- **Changes**:
  - Added `is_stereo=False` parameter to `AbstractAudioDataset.__init__()`
  - Parameter defaults to `False` for backwards compatibility
  - Passed through to instance variable `self.is_stereo`
- **Purpose**: Enable conditional stereo/mono processing in data loading

### Step 4: Modify Channel Averaging Logic
- **Status**: ✅ Completed
- **Files Modified**: `/sashimi/s4/src/dataloaders/audio.py`
- **Changes**:
  - Modified channel averaging logic in `__getitem__()` method
  - Changed from: `if seq.shape[0] > 1:` to `if seq.shape[0] > 1 and not self.is_stereo:`
  - Now preserves stereo channels when `is_stereo=True`
- **Purpose**: Enable stereo data loading while maintaining mono backwards compatibility

### Step 5: Update Tensor Reshaping Logic
- **Status**: ✅ Completed
- **Files Modified**: `/sashimi/s4/src/dataloaders/audio.py`
- **Changes**:
  - Updated tensor reshaping logic in `__getitem__()` method
  - Modified transpose operation to handle variable channel dimensions
  - Changed reshape from fixed `(1, L, 1)` to dynamic `(1, L, C)` where C=1 for mono, C=2 for stereo
- **Purpose**: Enable proper tensor shapes for both mono and stereo processing

### Step 6: Add is_stereo to QuantizedAutoregressiveAudio.init_defaults()
- **Status**: ✅ Completed
- **Files Modified**: `/sashimi/s4/src/dataloaders/audio.py`
- **Changes**:
  - Added `is_stereo: False` to `init_defaults()` method in `QuantizedAutoregressiveAudio` class
  - Ensures the parameter is available in dataset configurations
  - Maintains backwards compatibility with existing configs
- **Purpose**: Enable configuration-based stereo/mono mode selection

### Step 7: Make QuantizedAutoregressiveAudio.d_input Dynamic
- **Status**: ✅ Completed
- **Files Modified**: `/sashimi/s4/src/dataloaders/audio.py`
- **Changes**:
  - Modified `d_input` property to return 2 when `is_stereo=True`, 1 otherwise
  - Uses `getattr(self, 'is_stereo', False)` for safe attribute access
  - Enables automatic input dimension scaling based on stereo configuration
- **Purpose**: Allow models and downstream components to adapt to stereo/mono input dimensions

### Step 8: Pass is_stereo Parameter to Dataset Instances
- **Status**: ✅ Completed
- **Files Modified**: `/sashimi/s4/src/dataloaders/audio.py`
- **Changes**:
  - Modified `setup()` method in `QuantizedAutoregressiveAudio` class
  - Added `is_stereo=self.is_stereo` parameter to all dataset instance creations
  - Ensures stereo configuration is properly passed through to data loading
- **Purpose**: Enable dataset instances to respect stereo/mono configuration from YAML

### Step 9: Fix Stereo Embedding and Autoregressive Logic
- **Status**: ✅ Completed
- **Files Modified**:
  - `/sashimi/s4/src/dataloaders/audio.py` (autoregressive logic)
  - `/sashimi/s4/src/tasks/tasks.py` (embedding layer and normalization)
  - `/sashimi/s4/src/models/sequence/backbones/sashimi.py` (pooling dimensions)
  - `/sashimi/s4/train.py` (model instantiation with embedded dimension calculation)
- **Changes**:
  - **Data Loading**: Fixed autoregressive return to handle `(L, C)` shape for stereo
  - **Embedding**: Added `MultiChannelEmbedding` class for stereo inputs with concatenation
  - **Task**: Modified `LMTask` to handle multi-channel embedding with proper dimension scaling
  - **Normalization**: Fixed `ReversibleInstanceNorm1dInput` to use correct feature dimension after embedding
  - **Model**: Added `d_input` parameter to Sashimi backbone for proper pooling layer initialization
  - **Training**: Updated model instantiation to calculate embedded dimension (`d_model * channels`) for pooling layers
- **Purpose**: Resolve the einsum dimension mismatch by properly handling stereo inputs through the entire pipeline

### Next Steps
1. Implement joint channel processing with backwards compatibility
2. Start with adding `is_stereo` parameter to dataset configurations
3. Update data loading to conditionally preserve stereo channels
4. Modify model architecture to handle variable input dimensions
5. Test backwards compatibility with existing mono configurations
6. Validate cross-channel relationship modeling in stereo mode

---

