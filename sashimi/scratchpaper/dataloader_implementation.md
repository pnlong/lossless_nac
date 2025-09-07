# Chunked Training DataLoader Implementation Plan

## Overview
This implementation plan addresses the requirement to train on chunks of 2000 batches while cycling through the entire dataset (not just the first 2000 batches), with validation after each training chunk.

## Problem Analysis

### Current Behavior
- `limit_train_batches: 2000` and `limit_val_batches: 200` limit PyTorch DataLoader to yield only the first N batches per epoch
- This results in training only on a subset of the dataset, missing potentially valuable data

### Desired Behavior
- Process entire dataset in configurable chunks (e.g., 2000 training batches + 200 validation batches)
- Cycle through dataset completely, not just first chunks
- Maintain validation after each training chunk
- Track progress across training sessions

## Implementation Components

### 1. Custom ChunkedSampler
**Location**: `sashimi/s4/src/dataloaders/chunked_sampler.py`

**Purpose**: A PyTorch Sampler that yields dataset indices in configurable chunks, with automatic cycling.

**Key Features**:
- Cycles through entire dataset across multiple chunks
- Supports random shuffling within chunks
- Maintains state for resumability
- Configurable chunk sizes

**Implementation**:
```python
class ChunkedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset_size, chunk_size, seed=42):
        self.dataset_size = dataset_size
        self.chunk_size = chunk_size
        self.current_chunk = 0
        self.rng = np.random.RandomState(seed)

    def get_chunk_indices(self, chunk_idx):
        """Get indices for a specific chunk, cycling through dataset."""
        # Implementation details...

    def set_chunk(self, chunk_idx):
        """Set current chunk for iteration."""
        self.current_chunk = chunk_idx

    def __iter__(self):
        return iter(self.get_chunk_indices(self.current_chunk))

    def __len__(self):
        return min(self.chunk_size, self.dataset_size)
```

### 2. ChunkedDataLoader Wrapper
**Location**: `sashimi/s4/src/dataloaders/chunked_dataloader.py`

**Purpose**: Wrapper around PyTorch DataLoader that manages chunked iteration when enabled.

**Key Features**:
- **Automatically enabled** when `train_chunk_size` AND `val_chunk_size` are both specified
- Falls back to standard PyTorch DataLoader when chunk sizes not specified
- `train_chunk_size` takes precedence over `limit_train_batches` when both are present
- `val_chunk_size` takes precedence over `limit_val_batches` when both are present
- Automatic chunk progression when enabled
- Seamless integration with existing PyTorch Lightning training
- State management for training resumption
- Configurable chunk cycling behavior

**Implementation**:
```python
class ChunkedDataLoader:
    def __init__(self, dataset, batch_size, chunk_size=None, **dataloader_kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.dataloader_kwargs = dataloader_kwargs

        # Enable chunking if chunk_size is specified (for individual partition)
        self.use_chunked = chunk_size is not None

        if self.use_chunked:
            self.current_chunk = 0
            self.total_chunks = self._calculate_total_chunks()
        else:
            # Fallback to standard DataLoader
            self.standard_dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, **dataloader_kwargs
            )

    def _calculate_total_chunks(self):
        """Calculate how many chunks needed to cover entire dataset."""
        dataset_size = len(self.dataset)
        return (dataset_size + self.chunk_size - 1) // self.chunk_size

    def __iter__(self):
        """Return iterator - either chunked or standard."""
        if not self.use_chunked:
            return iter(self.standard_dataloader)

        # Chunked iteration
        sampler = ChunkedSampler(len(self.dataset), self.chunk_size)
        sampler.set_chunk(self.current_chunk)

        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            **self.dataloader_kwargs
        )

        # Progress to next chunk
        self.current_chunk = (self.current_chunk + 1) % self.total_chunks
        return iter(dataloader)

    def __len__(self):
        if not self.use_chunked:
            return len(self.standard_dataloader)
        return self.chunk_size
```

### 3. Enhanced SequenceDataset
**Location**: `sashimi/s4/src/dataloaders/base.py` (extend existing class)

**Purpose**: Add optional chunking support to the base dataset class while maintaining backward compatibility.

**Key Features**:
- **Chunked training enabled** when EITHER `train_chunk_size` OR `val_chunk_size` is specified
- When one chunk size is specified, the other is inferred to be the size of that partition
- Falls back to standard SequenceDataset behavior when no chunk sizes specified
- `train_chunk_size` takes precedence over `limit_train_batches`
- `val_chunk_size` takes precedence over `limit_val_batches`
- Integration with PyTorch Lightning

**Implementation**:
```python
class ChunkedSequenceDataset(SequenceDataset):
    def __init__(self, train_chunk_size=None, val_chunk_size=None, **kwargs):
        super().__init__(**kwargs)

        # Store explicit chunk sizes (take precedence over limit_* parameters)
        self.explicit_train_chunk_size = train_chunk_size
        self.explicit_val_chunk_size = val_chunk_size

        # Enable chunking if either chunk size is specified
        self.use_chunked_training = (train_chunk_size is not None or val_chunk_size is not None)

        # Infer missing chunk sizes from partition sizes when needed
        if self.use_chunked_training:
            self._infer_chunk_sizes(**kwargs)

        # Initialize chunk state if needed
        if self.use_chunked_training:
            self.current_train_chunk = 0
            self.current_val_chunk = 0

    def _infer_chunk_sizes(self, **kwargs):
        """Infer missing chunk sizes from partition sizes."""
        # If only one chunk size is specified, infer the other from partition size
        if self.explicit_train_chunk_size is not None and self.explicit_val_chunk_size is None:
            # Train chunking enabled, infer val chunk size from validation set size
            self.val_chunk_size = self._get_val_partition_size()
        elif self.explicit_val_chunk_size is not None and self.explicit_train_chunk_size is None:
            # Val chunking enabled, infer train chunk size from training set size
            self.train_chunk_size = self._get_train_partition_size()
        else:
            # Both explicitly specified
            self.train_chunk_size = self.explicit_train_chunk_size
            self.val_chunk_size = self.explicit_val_chunk_size

    def _get_train_partition_size(self):
        """Get the size of the training partition."""
        # Implementation to get training dataset size
        return len(self.dataset_train) if hasattr(self, 'dataset_train') else None

    def _get_val_partition_size(self):
        """Get the size of the validation partition."""
        # Implementation to get validation dataset size
        return len(self.dataset_val) if hasattr(self, 'dataset_val') else None

    def train_dataloader(self, **kwargs):
        """Return training dataloader - chunked or standard."""
        if not self.use_chunked_training or self.train_chunk_size is None:
            # Use standard SequenceDataset behavior (respect limit_train_batches)
            return super().train_dataloader(**kwargs)

        # Use chunked training dataloader (train_chunk_size takes precedence)
        return self._create_chunked_train_dataloader(**kwargs)

    def val_dataloader(self, **kwargs):
        """Return validation dataloader - chunked or standard."""
        if not self.use_chunked_training or self.val_chunk_size is None:
            # Use standard SequenceDataset behavior (respect limit_val_batches)
            return super().val_dataloader(**kwargs)

        # Use chunked validation dataloader (val_chunk_size takes precedence)
        return self._create_chunked_val_dataloader(**kwargs)

    def _create_chunked_train_dataloader(self, **kwargs):
        """Create chunked training dataloader."""
        # Implementation using self.train_chunk_size...

    def _create_chunked_val_dataloader(self, **kwargs):
        """Create chunked validation dataloader."""
        # Implementation using self.val_chunk_size...
```

### 4. Training Loop Modifications
**Location**: `sashimi/s4/train.py` (extend existing training)

**Purpose**: Conditionally modify training loop to handle chunked processing when enabled.

**Key Features**:
- **Automatically detects** chunked training when EITHER `train_chunk_size` OR `val_chunk_size` is specified
- Falls back to standard PyTorch Lightning training when no chunk sizes specified
- When one chunk size is specified, infers the other from partition size
- `train_chunk_size` takes precedence over `limit_train_batches`
- `val_chunk_size` takes precedence over `limit_val_batches`
- Progress tracking and logging when chunking is active
- Graceful handling of dataset boundaries

**Implementation**:
```python
class ChunkedTrainingMixin:
    def __init__(self, config):
        # Automatically detect chunking from config
        train_chunk_size = config.train.get('train_chunk_size')
        val_chunk_size = config.train.get('val_chunk_size')

        # Chunked training enabled if either chunk size is specified
        self.chunked_training_enabled = (train_chunk_size is not None or val_chunk_size is not None)

        # Store explicit chunk sizes
        self.explicit_train_chunk_size = train_chunk_size
        self.explicit_val_chunk_size = val_chunk_size

        if self.chunked_training_enabled:
            self.setup_chunked_training(config)

    def setup_chunked_training(self, config):
        """Initialize chunked training components."""
        # Infer missing chunk sizes from partition sizes
        self._infer_missing_chunk_sizes()

        # Replace standard dataset with chunked version
        # Configure chunk sizes (both will be set after inference)
        # Initialize progress tracking

    def _infer_missing_chunk_sizes(self):
        """Infer missing chunk sizes from partition sizes."""
        if self.explicit_train_chunk_size is not None and self.explicit_val_chunk_size is None:
            # Only train chunk size specified, infer val from validation partition
            self.val_chunk_size = self._get_val_partition_size()
            self.train_chunk_size = self.explicit_train_chunk_size
        elif self.explicit_val_chunk_size is not None and self.explicit_train_chunk_size is None:
            # Only val chunk size specified, infer train from training partition
            self.train_chunk_size = self._get_train_partition_size()
            self.val_chunk_size = self.explicit_val_chunk_size
        else:
            # Both specified explicitly
            self.train_chunk_size = self.explicit_train_chunk_size
            self.val_chunk_size = self.explicit_val_chunk_size

    def _get_train_partition_size(self):
        """Get training partition size for inference."""
        # Implementation to get training dataset size
        pass

    def _get_val_partition_size(self):
        """Get validation partition size for inference."""
        # Implementation to get validation dataset size
        pass

    def train_epoch_with_chunking(self):
        """Single training epoch with chunked processing."""
        if not self.chunked_training_enabled:
            # Use standard training loop
            return self.standard_training_epoch()

        # Both partitions will use chunking (one explicit, one inferred)
        # Train on chunk (using self.train_chunk_size)
        # Validate on chunk (using self.val_chunk_size)
        # Update progress

    def standard_training_epoch(self):
        """Standard training epoch - used when no chunk sizes specified."""
        # Standard PyTorch Lightning training loop
        pass
```

### 5. Audio Dataset Integration
**Location**: `sashimi/s4/src/dataloaders/audio.py` (extend existing classes)

**Purpose**: Add chunking support to audio-specific datasets.

**Key Features**:
- Integration with existing audio preprocessing
- Support for different audio formats and sampling rates
- Maintain audio-specific collate functions
- Preserve quantization and transformation pipelines

**Implementation**:
```python
class ChunkedQuantizedAudioDataset(QuantizedAudioDataset):
    """Audio dataset with chunked training support."""

class ChunkedQuantizedAutoregressiveAudio(QuantizedAutoregressiveAudio):
    """Audio autoregressive dataset with chunking."""
```

## Configuration Updates

### YAML Configuration
**Location**: `sashimi/s4/configs/experiment/audio/sashimi-musdb18mono.yaml`

**Changes**:
```yaml
# Full chunked training (both partitions explicitly specified):
train:
  train_chunk_size: 2000               # Training batches per chunk
  val_chunk_size: 200                  # Validation batches per chunk
  # Note: limit_train_batches and limit_val_batches are IGNORED when chunk sizes are specified

loader:
  batch_size: 32  # Actual batch size for DataLoader

# Chunked training with inference (only train_chunk_size specified):
train:
  train_chunk_size: 2000               # Training uses chunking
  # val_chunk_size will be inferred from validation partition size
  limit_val_batches: 200               # IGNORED - val_chunk_size will be inferred

loader:
  batch_size: 32

# Chunked validation with inference (only val_chunk_size specified):
train:
  val_chunk_size: 200                  # Validation uses chunking
  # train_chunk_size will be inferred from training partition size
  limit_train_batches: 2000            # IGNORED - train_chunk_size will be inferred

loader:
  batch_size: 32

# Standard training (existing behavior - no changes needed):
train:
  # No chunk_size parameters specified = standard behavior
  limit_train_batches: 2000            # This will be used (standard behavior)
  limit_val_batches: 200               # This will be used (standard behavior)

loader:
  batch_size: 32  # Existing batch size
```

### Backward Compatibility
**Zero Configuration Changes Required:**
- Existing configurations continue to work **unchanged**
- New chunking features are **automatically enabled** when chunk size parameters are present
- All existing YAML files work without modification
- `limit_train_batches` and `limit_val_batches` are respected when no chunk sizes specified
- **Inference**: When only one chunk size is specified, the other is automatically inferred
- **Precedence**: `train_chunk_size` > `limit_train_batches`, `val_chunk_size` > `limit_val_batches`

**Migration Path:**
```yaml
# Existing config (continues to work unchanged):
trainer:
  limit_train_batches: 2000
  limit_val_batches: 200

# New config (full chunked training):
trainer:
  limit_train_batches: 2000  # IGNORED when train_chunk_size specified
  limit_val_batches: 200     # IGNORED when val_chunk_size specified
train:
  train_chunk_size: 2000     # Takes precedence over limit_train_batches
  val_chunk_size: 200        # Takes precedence over limit_val_batches

# New config (chunked training with inference):
trainer:
  limit_train_batches: 2000  # IGNORED when train_chunk_size specified
  limit_val_batches: 200     # IGNORED - val_chunk_size will be inferred
train:
  train_chunk_size: 2000     # Takes precedence over limit_train_batches
  # val_chunk_size automatically inferred from validation partition size

# New config (chunked validation with inference):
trainer:
  limit_train_batches: 2000  # IGNORED - train_chunk_size will be inferred
  limit_val_batches: 200     # IGNORED when val_chunk_size specified
train:
  val_chunk_size: 200        # Takes precedence over limit_val_batches
  # train_chunk_size automatically inferred from training partition size
```

## Backward Compatibility Guarantee

### Critical Requirements
**The implementation MUST ensure that:**

1. **Existing YAML configurations work unchanged**
   - No modifications required to existing config files
   - `limit_train_batches` and `limit_val_batches` continue to work when no chunk sizes specified
   - All existing training scripts run without modification

2. **Parameter precedence rules**
   - `train_chunk_size` takes precedence over `limit_train_batches` when both are present
   - `val_chunk_size` takes precedence over `limit_val_batches` when both are present
   - `limit_train_batches` and `limit_val_batches` are ignored when chunk sizes are specified

3. **Automatic detection and inference**
   - **Chunked training**: enabled when EITHER `train_chunk_size` OR `val_chunk_size` is specified
   - **Inference**: When only one chunk size is specified, the other is automatically inferred from partition size
   - **Standard training**: used when neither chunk size is specified
   - No performance impact when chunk sizes are not specified

4. **Flexible partition control**
   - Specify only `train_chunk_size`: enables chunked training, infers val chunk size from validation partition
   - Specify only `val_chunk_size`: enables chunked validation, infers train chunk size from training partition
   - Specify both: uses explicit chunk sizes for both partitions
   - Specify neither: uses standard behavior with `limit_train_batches` and `limit_val_batches`

### Implementation Guards
Each component must include automatic detection and precedence handling:

```python
# Example in ChunkedDataLoader
def __init__(self, chunk_size=None, limit_batches=None, **kwargs):
    # Precedence: chunk_size takes priority over limit_batches
    self.effective_chunk_size = chunk_size if chunk_size is not None else limit_batches
    self.use_chunked = self.effective_chunk_size is not None  # Automatic detection

    if self.use_chunked:
        # Initialize chunked components
        self._setup_chunked_mode()
    else:
        # Use standard PyTorch DataLoader
        self._setup_standard_mode()

# Example in dataset classes
def __init__(self, train_chunk_size=None, val_chunk_size=None, **kwargs):
    # Store explicit chunk sizes
    self.explicit_train_chunk_size = train_chunk_size
    self.explicit_val_chunk_size = val_chunk_size

    # Enable chunking if either is specified
    self.use_chunked_training = (train_chunk_size is not None or val_chunk_size is not None)

    if self.use_chunked_training:
        self._infer_missing_chunk_sizes(**kwargs)

def _infer_missing_chunk_sizes(self, **kwargs):
    """Infer missing chunk sizes from partition sizes."""
    if self.explicit_train_chunk_size is not None and self.explicit_val_chunk_size is None:
        # Train chunking enabled, infer val chunk size from validation partition
        self.val_chunk_size = self._get_val_partition_size()
        self.train_chunk_size = self.explicit_train_chunk_size
    elif self.explicit_val_chunk_size is not None and self.explicit_train_chunk_size is None:
        # Val chunking enabled, infer train chunk size from training partition
        self.train_chunk_size = self._get_train_partition_size()
        self.val_chunk_size = self.explicit_val_chunk_size
    else:
        # Both explicitly specified
        self.train_chunk_size = self.explicit_train_chunk_size
        self.val_chunk_size = self.explicit_val_chunk_size

def train_dataloader(self, **kwargs):
    if self.use_chunked_training and self.train_chunk_size is not None:
        return self._create_chunked_train_dataloader(**kwargs)
    else:
        return super().train_dataloader(**kwargs)  # Standard behavior
```

### Testing Requirements
- **Regression tests**: Ensure existing functionality works when no chunk sizes specified
- **Configuration tests**: Test various config combinations:
  - No chunk sizes (standard behavior with limit_* parameters)
  - Only `train_chunk_size` specified (inference mode - val chunk size inferred)
  - Only `val_chunk_size` specified (inference mode - train chunk size inferred)
  - Both chunk sizes specified (explicit mode)
- **Precedence tests**: Verify chunk_size parameters override limit_* parameters
  - `train_chunk_size` takes precedence over `limit_train_batches`
  - `val_chunk_size` takes precedence over `limit_val_batches`
  - `limit_*` parameters ignored when chunk_size parameters present
- **Automatic detection tests**: Verify chunking modes are detected correctly
- **Inference tests**: Verify chunk sizes are properly inferred from partition sizes
  - Test that val chunk size is inferred when only train_chunk_size is specified
  - Test that train chunk size is inferred when only val_chunk_size is specified
  - Test that explicit chunk sizes are used when both are specified
- **Performance tests**: Verify no performance degradation when chunking disabled
- **Backward compatibility tests**: Ensure existing YAML configs work unchanged

## Implementation Phases

### Phase 1: Core Components (Week 1)
1. Implement `ChunkedSampler`
2. Implement `ChunkedDataLoader`
3. Create basic tests for core functionality
4. **Verify automatic detection**: Test that chunking is enabled when chunk sizes are specified
5. **Verify precedence rules**: Test that chunk_size parameters override limit_* parameters
6. **Verify backward compatibility**: Ensure existing code works when no chunk sizes are specified
7. **Verify inference behavior**: Test that chunk sizes are properly inferred when only one is specified
8. Test fallback behavior and mixed partition configurations

### Phase 2: Dataset Integration (Week 2)
1. Extend `SequenceDataset` with chunking support
2. Update audio datasets to inherit chunking behavior
3. Test with existing audio datasets

### Phase 3: Training Loop Integration (Week 3)
1. Modify training loop to support chunked iteration
2. Add progress tracking and logging
3. Implement validation after each chunk

### Phase 4: Testing and Optimization (Week 4)
1. Comprehensive testing with different dataset sizes
2. Performance optimization
3. Memory usage analysis
4. Integration testing with PyTorch Lightning

## Benefits

1. **Complete Dataset Utilization**: Cycles through entire dataset, not just first chunks
2. **Flexible Configuration**: Easily adjustable chunk sizes via config
3. **Seamless Integration**: Minimal changes to existing codebase
4. **Resume Capability**: Can resume training from specific chunks
5. **Validation Integration**: Automatic validation after each training chunk
6. **Scalability**: Works with datasets of any size

## Testing Strategy

### Unit Tests
- Test `ChunkedSampler` with different chunk sizes
- Test `ChunkedDataLoader` iteration behavior
- Test chunk boundary handling

### Integration Tests
- Test with actual audio datasets
- Verify training loop integration
- Test configuration loading

### Performance Tests
- Compare training time with standard vs chunked approach
- Memory usage analysis
- Scalability testing with large datasets

## Risk Mitigation

1. **Backward Compatibility**: **CRITICAL** - Ensure existing code continues to work unchanged
   - Implement comprehensive fallback mechanisms
   - Test with existing configurations before each phase
   - Maintain identical behavior when chunked training disabled
   - Zero configuration changes required for existing users

2. **Parameter Precedence**: **CRITICAL** - Ensure `train_chunk_size` and `val_chunk_size` properly override `limit_*` parameters
   - Implement clear precedence logic in all components
   - Test precedence rules extensively
   - Document precedence behavior clearly
   - Prevent conflicts between old and new parameters

3. **Automatic Detection**: **HIGH** - Ensure chunking modes are detected correctly
   - Test all parameter combinations thoroughly
   - Verify fallback to standard behavior works
   - Validate mixed mode configurations

4. **Inference Logic**: **HIGH** - Ensure chunk sizes are properly inferred from partition sizes
   - Test that missing chunk sizes are correctly inferred
   - Verify inference works for both train and validation partitions
   - Validate that explicit chunk sizes take precedence over inferred ones

5. **Memory Usage**: Monitor for potential memory leaks with large chunks
6. **Performance**: Benchmark against standard training approach
7. **State Management**: Robust handling of training state and resumption

## Success Metrics

1. **Backward Compatibility**: **CRITICAL** - All existing configurations work unchanged
2. **Parameter Precedence**: **CRITICAL** - `train_chunk_size` > `limit_train_batches`, `val_chunk_size` > `limit_val_batches`
3. **Automatic Detection**: Successfully detects and enables appropriate chunking mode based on parameters
4. **Inference Logic**: Successfully infers missing chunk sizes from partition sizes
5. **Functionality**: Successfully trains on entire dataset in chunks when enabled
6. **Performance**: No significant performance degradation (with or without chunking)
7. **Usability**: Easy opt-in configuration and seamless integration
8. **Reliability**: Robust error handling and state management

## Future Enhancements

1. **Dynamic Chunk Sizing**: Adjust chunk sizes based on training progress
2. **Intelligent Sampling**: Prioritize certain data chunks based on learning objectives
3. **Parallel Chunk Processing**: Process multiple chunks simultaneously
4. **Adaptive Validation**: Adjust validation frequency based on training dynamics
