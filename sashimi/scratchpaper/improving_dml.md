# DML Training Issues Analysis & Solutions

## Problem Summary

The DML models are experiencing two critical issues:
1. **Training Failure**: Loss is not decreasing and hovers around starting values
2. **Memory Usage**: DML models consume more GPU memory than expected despite having fewer parameters

## Root Cause Analysis

### 1. Training Issues - Why Loss Isn't Decreasing

#### A. **Gradient Scaling Problems**
The DML loss function has several numerical stability issues that can cause gradient problems:

**Issue**: The loss computation involves multiple exponential operations and log-space calculations that can lead to:
- **Gradient Vanishing**: `torch.exp(-log_scales)` and `torch.sigmoid()` operations can produce very small gradients
- **Gradient Explosion**: `torch.exp(log_scales)` can produce very large values when `log_scales` is not properly clamped
- **Numerical Instability**: The complex nested `torch.where` conditions can create discontinuous gradients

**Evidence**: Looking at the loss function (lines 166-200), there are multiple exponential operations:
```python
inv_scales = torch.exp(-log_scales)  # Can be very large if log_scales is negative
cdf_plus = torch.sigmoid(plus_in)   # Can saturate and kill gradients
log_cdf_plus = plus_in - F.softplus(plus_in)  # Complex log-space computation
```

#### B. **Learning Rate Mismatch**
The current learning rate (0.004) may be inappropriate for DML training:

**Issue**: DML loss has different gradient magnitudes compared to categorical cross-entropy:
- Categorical loss: Direct logit gradients
- DML loss: Complex mixture parameter gradients with exponential scaling

**Evidence**: The loss involves mixture weights, means, and log-scales that have different gradient scales.

#### C. **Initialization Problems**
The DML output head parameters may not be properly initialized:

**Issue**: The output head uses raw linear projections without proper initialization:
```python
# In output_heads.py lines 60-61
# No activation on logits - will use log_softmax in loss
# No activation on means - can be any real value  
# No activation on log_scales - exponential in loss for positivity
```

**Problem**: Without proper initialization, the mixture components may start in poor configurations.

### 2. Memory Usage Issues - Why DML Uses More Memory

#### A. **Your Understanding is CORRECT - But There's More**

**Your Calculation**: 
- Categorical: `2^bits` outputs (256 for 8-bit, 65536 for 16-bit)
- DML: `3 * n_mixtures` outputs (18 for 6 mixtures, 15 for 5 mixtures)

**You're absolutely right** - DML should use much less memory for the output head. However, there are additional factors:

#### B. **Stereo Processing Overhead**
The stereo configurations double memory usage:

**Evidence**: From the configs:
- `musdb18mono.yaml`: `bits: 16`, `sample_len: 8192`
- `musdb18stereo.yaml`: `bits: 16`, `sample_len: 12288`, `is_stereo: true`

**Memory Impact**: 
- Stereo doubles the sequence length: `12288 * 2 = 24576` vs `8192`
- DML stereo: `(batch, 24576, 3*5) = (batch, 24576, 15)` vs categorical: `(batch, 24576, 65536)`

#### C. **Loss Function Memory Overhead**
The DML loss function creates many intermediate tensors:

**Memory Intensive Operations**:
```python
# Lines 148-153: Stereo reshaping creates copies
logit_probs = logit_probs.view(logit_probs.shape[0], -1, logit_probs.shape[-1])
means = means.view(means.shape[0], -1, means.shape[-1]) 
log_scales = log_scales.view(log_scales.shape[0], -1, log_scales.shape[-1])

# Lines 175-200: Multiple intermediate tensors
x_plus = x + bin_size / 2
x_minus = x - bin_size / 2
centered_x = x - means
plus_in = inv_scales * (centered_x + bin_size / 2)
min_in = inv_scales * (centered_x - bin_size / 2)
cdf_plus = torch.sigmoid(plus_in)
cdf_min = torch.sigmoid(min_in)
cdf_delta = cdf_plus - cdf_min
```

**Memory Impact**: Each operation creates new tensors, multiplying memory usage.

#### D. **SaShiMi Architecture Overhead**
The SaShiMi model itself has significant memory overhead:

**From `gpu_memory_optimizations.md`**:
- U-Net skip connections: ~2-4x sequence length × d_model per down block
- Multiple layer states: ~n_layers × batch_size × d_model
- Pooling operations create additional intermediate tensors

## Solutions

### 1. Fix Training Issues

#### A. **Improve Gradient Stability**
```python
# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Improve log_scales clamping
log_scales = torch.clamp(log_scales, min=-5.0, max=5.0)  # More conservative than -7.0
```

#### B. **Adjust Learning Rate**
```yaml
# In experiment configs
optimizer:
  lr: 0.001  # Reduce from 0.004 for DML
```

#### C. **Better Initialization**
```python
# In DiscretizedLogisticMixtureHead.__init__
def __init__(self, d_model, n_mixtures):
    super().__init__()
    self.n_mixtures = n_mixtures
    self.projection = nn.Linear(d_model, 3 * n_mixtures)
    
    # Better initialization
    with torch.no_grad():
        # Initialize means to small values
        self.projection.weight[1*n_mixtures:2*n_mixtures].fill_(0.0)
        # Initialize log_scales to reasonable values
        self.projection.weight[2*n_mixtures:3*n_mixtures].fill_(-1.0)
```

### 2. Fix Memory Issues

#### A. **Reduce Sequence Length for Testing**
```yaml
# In dataset configs
sample_len: 4096  # Reduce from 8192/12288 for testing
```

#### B. **Optimize Loss Function**
```python
# Use in-place operations where possible
log_scales = torch.clamp_(log_scales, min=-7.0)  # In-place
inv_scales = torch.exp_(-log_scales)  # In-place
```

#### C. **Reduce Batch Size**
```yaml
# In experiment configs
loader:
  batch_size: 1  # Already at minimum, but ensure this is set
```

### 3. Debugging Steps

#### A. **Monitor Gradient Norms**
```python
# Add to training loop
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** (1. / 2)
print(f"Gradient norm: {total_norm}")
```

#### B. **Monitor Loss Components**
```python
# Add to DML loss function
print(f"avg_scale: {avg_scale}, avg_mean: {avg_mean}, mixture_entropy: {mixture_entropy}")
```

#### C. **Test with Smaller Model**
```yaml
# Reduce model size for debugging
model:
  d_model: 32  # Reduce from 64
  n_layers: 4  # Reduce from 8
  n_mixtures: 3  # Reduce from 5/6
```

## Expected Outcomes

After implementing these fixes:

1. **Training**: Loss should start decreasing within first few epochs
2. **Memory**: Should use significantly less memory than categorical models
3. **Performance**: Should achieve better compression (lower BPB) than categorical models

## ✅ IMPLEMENTED SOLUTIONS

### 1. **Improved DML Output Head Initialization**
**File**: `src/models/sequence/modules/output_heads.py`

Added proper initialization for DML parameters:
- **Mixture weights**: Small random values (σ=0.1) to break symmetry
- **Means**: Small values around zero (σ=0.1) 
- **Log-scales**: Reasonable starting values (-1.0 to -0.5) giving scales ~0.37-0.61

```python
def _init_parameters(self):
    with torch.no_grad():
        # Initialize mixture weights to be roughly uniform
        self.projection.weight[:self.n_mixtures].normal_(0, 0.1)
        # Initialize means to small values around zero  
        self.projection.weight[self.n_mixtures:2*self.n_mixtures].normal_(0, 0.1)
        # Initialize log_scales to reasonable values (-1.0 to -0.5)
        self.projection.bias[2*self.n_mixtures:3*self.n_mixtures].uniform_(-1.0, -0.5)
```

### 2. **Gradient Clipping & Monitoring**
**File**: `train.py`

Added gradient clipping and monitoring for DML models:
- **Gradient clipping**: `max_norm=1.0` to prevent explosion
- **Gradient norm logging**: Monitor `trainer/grad_norm` in WandB
- **DML-specific**: Only applies to models with `output_head_type == 'dml'`

```python
if hasattr(self.model, 'output_head_type') and self.model.output_head_type == 'dml':
    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
    # Log gradient norm for monitoring
    self.log("trainer/grad_norm", total_norm, ...)
```

### 3. **Improved Loss Function Stability**
**File**: `src/models/sequence/loss.py`

Enhanced numerical stability:
- **Conservative clamping**: `log_scales` clamped to `[-5.0, 5.0]` instead of `[-7.0, inf]`
- **Memory optimization**: Added comments for in-place operations
- **Better gradient flow**: More conservative bounds prevent gradient issues

### 7. **Fixed Validation Chunk Size Bug**
**File**: `src/dataloaders/base.py`

Fixed critical bug where validation dataloaders were incorrectly using `train_chunk_size` instead of `val_chunk_size`:

**The Problem**: 
- `_eval_dataloader()` set `_current_dataloader_type = 'val'` or `'test'`
- But chunking logic only checked for `_current_dataloader_type == 'eval'`
- This caused validation to fall back to shuffle-based detection, using wrong chunk size

**The Fix**:
```python
# Before: Only checked for 'eval' type
elif dataloader_type == 'eval':
    return self.use_val_chunking

# After: Check for both 'val' and 'test' types
elif dataloader_type in ['val', 'test']:
    return self.use_val_chunking
```

This ensures validation dataloaders correctly use `val_chunk_size: 200` instead of `train_chunk_size: 2000`.

## Next Steps

1. ✅ ~~Implement gradient clipping and better initialization~~
2. ✅ ~~Reduce learning rate and test with smaller sequences~~
3. ✅ ~~Monitor gradient norms and loss components~~
4. ✅ ~~Fix validation chunk size bug~~
5. **Test the optimizations**: Run training with debug config first
6. **Compare memory usage**: Monitor GPU memory with optimizations
7. **Validate training**: Check if loss decreases properly
8. **Scale up**: Test with full-size models once debug works

## Key Insight

Your understanding of DML parameter count is **correct** - DML should use much less memory. The issue is that the current implementation has:
1. **Training instability** preventing proper learning
2. **Stereo processing overhead** doubling memory usage
3. **Loss function inefficiency** creating unnecessary intermediate tensors

The memory issue is likely masking the training issue - the model isn't learning properly, so it's not optimizing memory usage either.
