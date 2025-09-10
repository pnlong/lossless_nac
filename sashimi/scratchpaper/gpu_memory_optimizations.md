# GPU Memory Optimization Analysis

## Current Memory Usage Analysis

Based on the GPU memory test results, the system successfully allocates up to **18.16 GiB** before failing, but your original training script fails when trying to allocate **1.50 GiB** with only **1.22 GiB** free. This indicates that the training process is consuming approximately **22 GiB** of GPU memory, leaving insufficient space for additional allocations.

## Identified Memory Bottlenecks

### 1. **SaShiMi Model Architecture** (Major Contributor)
**Location**: `src/models/sequence/backbones/sashimi.py`

**Memory Issues**:
- **U-Net Skip Connections**: The model stores intermediate outputs from down blocks in `outputs = []` list (lines 247-248, 256, 264-265)
- **Multiple Layer States**: Each layer maintains its own state, and the model creates states for all layers: `[layer.default_state(*args, **kwargs) for layer in layers]` (line 291)
- **Pooling Operations**: DownPool and UpPool operations create additional intermediate tensors
- **DML Output Head**: The DML head outputs 3×n_mixtures parameters, creating larger tensors than categorical output

**Memory Impact**: 
- Skip connections: ~2-4x sequence length × d_model per down block
- Layer states: ~n_layers × batch_size × d_model
- DML head: 3×n_mixtures × sequence_length × batch_size

### 2. **DML Output Head** (Significant Contributor)
**Location**: `src/models/sequence/modules/output_heads.py`

**Memory Issues**:
- **Triple Output**: DML head outputs 3×n_mixtures parameters (logit_probs, means, log_scales) vs single categorical output
- **Dictionary Storage**: Returns dictionary with multiple tensors instead of single tensor
- **Loss Computation**: DML loss function processes mixture parameters, requiring additional memory for softmax computations

**Memory Impact**:
- Output tensor: 3×n_mixtures × sequence_length × batch_size × 4 bytes (float32)
- For n_mixtures=6, bits=8: 3×6×seq_len×batch_size = 18×seq_len×batch_size vs 256×seq_len×batch_size for categorical

### 3. **Audio Dataset Processing** (Moderate Contributor)
**Location**: `src/dataloaders/audio.py`

**Memory Issues**:
- **Sequence Padding**: `collate_fn` pads sequences to power-of-2 lengths (lines 432-434)
- **Stereo Processing**: Interleaving doubles sequence length (lines 167-178)
- **Quantization**: Creates additional tensor copies during quantization process
- **Resampling**: Caches resampling transforms per sample rate (lines 160-162)

**Memory Impact**:
- Padding: Can increase sequence length by up to 2x
- Stereo: Doubles sequence length (L → L×2)
- Quantization: Creates temporary tensors during encoding/decoding

### 4. **Training Loop State Management** (Moderate Contributor)
**Location**: `train.py`

**Memory Issues**:
- **State Persistence**: `_state` maintained across batches (lines 415, 463)
- **Memory Chunks**: BPTT mode stores previous batches in `_memory_chunks` (lines 451-455)
- **Gradient Accumulation**: PyTorch Lightning may accumulate gradients
- **EMA Weights**: Exponential moving average creates duplicate model parameters (lines 711-718)

**Memory Impact**:
- State: ~n_layers × batch_size × d_model
- Memory chunks: ~n_context × batch_size × sequence_length × d_model
- EMA: ~2x model parameters

### 5. **Encoder/Decoder Processing** (Minor Contributor)
**Location**: `src/tasks/encoders.py`, `src/tasks/decoders.py`

**Memory Issues**:
- **Embedding Lookup**: Large embedding tables for high-bit quantization (2^bits entries)
- **Sequence Decoder**: Maintains full sequence length throughout processing
- **Passthrough Sequential**: May create intermediate tensor copies

**Memory Impact**:
- Embedding: 2^bits × d_model × 4 bytes
- For bits=8: 256 × d_model × 4 bytes
- Sequence processing: Full sequence length maintained

## Memory Usage Estimation

### Current Configuration Analysis
- **Model**: SaShiMi with d_model=64, n_layers=8, n_mixtures=6
- **Dataset**: 8-bit audio, stereo processing
- **Sequence Length**: Variable (depends on audio files)

### Estimated Memory Breakdown
1. **Model Parameters**: ~50-100 MB
2. **Skip Connections**: ~2-4 GB (depends on sequence length)
3. **Layer States**: ~500 MB - 1 GB
4. **DML Output**: ~1-2 GB (3×6×seq_len×batch_size)
5. **Audio Processing**: ~1-2 GB (padding, stereo, quantization)
6. **Training State**: ~500 MB - 1 GB
7. **PyTorch Overhead**: ~2-3 GB

**Total Estimated**: ~8-12 GB (but actual usage is ~22 GB, indicating additional overhead)

## Recommended Optimizations

### High Impact Optimizations

1. **Reduce Model Size**
   - Decrease `n_layers` from 8 to 6
   - Reduce `n_mixtures` from 6 to 4
   - Lower `d_model` if possible

2. **Enable Gradient Checkpointing**
   - Add `gradient_checkpointing: true` to model config
   - Trade computation for memory

3. **Optimize Skip Connections**
   - Implement selective skip connection storage
   - Use in-place operations where possible

4. **Reduce Sequence Length**
   - Decrease `sample_len` in dataset config
   - Use shorter audio chunks

### Medium Impact Optimizations

5. **Memory-Efficient DML**
   - Process mixture parameters in chunks
   - Use mixed precision training

6. **Optimize Data Loading**
   - Reduce `batch_size` to 1 (already done)
   - Use `num_workers=0` to reduce memory overhead
   - Implement streaming data loading

7. **State Management**
   - Clear states more frequently
   - Use `torch.cuda.empty_cache()` strategically

### Low Impact Optimizations

8. **PyTorch Memory Settings**
   - Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`
   - Enable memory fragmentation reduction

9. **Audio Processing**
   - Use more efficient quantization
   - Reduce padding overhead
   - Optimize stereo processing

## Implementation Priority

1. **Immediate** (Easy, High Impact):
   - Reduce model parameters (n_layers, n_mixtures)
   - Enable gradient checkpointing
   - Set PyTorch memory configuration

2. **Short-term** (Medium Effort, High Impact):
   - Optimize skip connection storage
   - Implement memory-efficient DML processing
   - Reduce sequence length

3. **Long-term** (High Effort, Medium Impact):
   - Rewrite audio processing pipeline
   - Implement custom memory management
   - Optimize state handling

## Expected Memory Reduction

- **Model size reduction**: 20-30% reduction
- **Gradient checkpointing**: 30-50% reduction
- **Skip connection optimization**: 10-20% reduction
- **Sequence length reduction**: 20-40% reduction

**Total Expected Reduction**: 50-70% of current memory usage, bringing it down to ~7-11 GB, which should fit comfortably within your 23.69 GiB GPU memory.
