# DML Implementation Bug Fixes

## Overview
This document identifies critical issues in the discretized mixture logistic (DML) implementation that are causing training instability and preventing convergence. The main symptoms are:
- Loss only decreases for ~100 steps then oscillates without converging
- Loss spikes from ~5.5 to >400 when learning rate is increased from 0.004 to 0.1
- Models don't learn effectively despite having correct architecture

## Critical Issues Identified

### 1. **Numerical Instability in Loss Function** 
**File**: `sashimi/s4/src/models/sequence/loss.py`
**Lines**: 166-201

**Problem**: Multiple exponential operations create numerical instability:
```python
# Line 170: Can produce very large values
inv_scales = torch.exp(-log_scales)

# Line 188-189: Can saturate and kill gradients  
cdf_plus = torch.sigmoid(plus_in)
cdf_min = torch.sigmoid(min_in)

# Line 194-201: Complex log-space computations prone to overflow
log_cdf_plus = plus_in - F.softplus(plus_in)
log_one_minus_cdf_min = -F.softplus(min_in)
log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)
```

**Impact**: 
- Gradient vanishing when sigmoid saturates
- Gradient explosion when exp(-log_scales) becomes very large
- Loss spikes when numerical overflow occurs

**Solution**: 
- Add more conservative clamping: `log_scales = torch.clamp(log_scales, min=-5.0, max=5.0)`
- Use numerically stable sigmoid implementation
- Add gradient clipping in training loop

### 2. **Incorrect Bin Size Calculation**
**File**: `sashimi/s4/src/models/sequence/loss.py`
**Line**: 174

**Problem**: Bin size calculation doesn't match the original PixelCNN++ implementation:
```python
# Current (INCORRECT):
bin_size = 1.0 / (num_classes - 1)  # For 8-bit: 1/255

# Should be (CORRECT):
bin_size = 1.0 / (num_classes - 1)  # For 8-bit: 1/255 (matches original)
```

**Impact**: 
- Actually, the current implementation is CORRECT for the bin size calculation
- The issue is elsewhere in the implementation
- This was a false alarm in the initial analysis

**Solution**: No change needed - the bin size calculation is correct

### 3. **Edge Case Handling Issues**
**File**: `sashimi/s4/src/models/sequence/loss.py`
**Lines**: 211-223

**Problem**: Edge case thresholds (-0.999, 0.999) are too restrictive and don't properly handle quantization boundaries:
```python
# Current thresholds are too narrow
log_probs = torch.where(
    x < -0.999,  # Too restrictive
    log_cdf_plus,
    torch.where(
        x > 0.999,  # Too restrictive  
        log_one_minus_cdf_min,
        # ...
    )
)
```

**Impact**:
- Many valid quantized values fall into the complex middle case
- Inconsistent gradient behavior near boundaries
- Loss computation errors for edge cases

**Solution**: Use proper quantization-aware edge detection:
```python
is_left_edge = (targets == 0).unsqueeze(-1)
is_right_edge = (targets == num_classes - 1).unsqueeze(-1)
```

### 4. **Mixture Weight Computation Error**
**File**: `sashimi/s4/src/models/sequence/loss.py`
**Lines**: 225-230

**Problem**: The mixture weight computation is mathematically correct but computationally inefficient and can cause gradient issues:
```python
# Current approach is correct but inefficient
mixture_log_probs = log_prob_from_logits(logit_probs)  # Custom function
log_probs = log_probs + mixture_log_probs
log_probs = torch.logsumexp(log_probs, dim=-1)
```

**Impact**:
- Unnecessary computational overhead
- Potential gradient flow issues with custom log_softmax
- Inconsistent with PyTorch's optimized implementations

**Solution**: Use PyTorch's optimized log_softmax:
```python
log_probs = F.log_softmax(logit_probs, dim=-1) + torch.log(bin_prob)
log_probs = torch.logsumexp(log_probs, dim=-1)
```

### 5. **Log Scales Clamping Too Permissive**
**File**: `sashimi/s4/src/models/sequence/loss.py`
**Line**: 167

**Problem**: Current clamping allows extreme values that cause numerical issues:
```python
# Current: Only clamps minimum, allows very large positive values
log_scales = torch.clamp(log_scales, min=-7.0)
```

**Impact**:
- `exp(-log_scales)` can become extremely large when log_scales is very negative
- `exp(log_scales)` can become extremely large when log_scales is very positive
- Gradient explosion and loss spikes

**Solution**: Use symmetric clamping:
```python
log_scales = torch.clamp(log_scales, min=-5.0, max=5.0)
```

### 6. **Sampling Function Numerical Issues**
**File**: `sashimi/s4/src/models/sequence/loss.py`
**Lines**: 44-45, 68-69

**Problem**: Sampling function has similar numerical stability issues:
```python
# Line 45: Same clamping issue as loss function
log_scales = torch.clamp(log_scales, min=-7.0)

# Line 68-69: Can produce extreme values
u = torch.rand_like(selected_means) * 0.999 + 1e-5
logistic_samples = selected_means + torch.exp(selected_log_scales) * (torch.log(u) - torch.log(1. - u))
```

**Impact**:
- Inconsistent sampling behavior
- Potential overflow during generation
- Mismatch between training and inference

**Solution**: Apply same clamping fixes as loss function

### 7. **Learning Rate Mismatch**
**File**: Training configuration files

**Problem**: DML loss has different gradient magnitudes than categorical cross-entropy, but uses same learning rate:
- Categorical: Direct logit gradients
- DML: Complex mixture parameter gradients with exponential scaling

**Impact**:
- Learning rate too high causes gradient explosion (loss spike to >400)
- Learning rate too low causes slow convergence
- No convergence after initial ~100 steps

**Solution**: 
- Use lower learning rate for DML: `lr: 0.001` instead of `0.004`
- Add learning rate scheduling
- Implement gradient clipping: `max_norm=1.0`

### 8. **Missing Gradient Monitoring**
**File**: `sashimi/s4/train.py`

**Problem**: No gradient norm monitoring to detect explosion/vanishing:
- Can't detect when gradients become too large/small
- No early warning for training instability
- Difficult to debug convergence issues

**Solution**: Add gradient norm logging (already implemented in lines 664-673)

## Priority Fixes

### High Priority (Fix First)
1. **Add symmetric log_scales clamping** (Line 167) - Prevents gradient explosion
2. **Implement proper edge case handling** (Lines 211-223) - Fixes boundary issues
3. **Add gradient clipping** (Training loop) - Prevents loss spikes
4. **Fix mixture weight computation** (Lines 225-230) - Improves efficiency and stability

### Medium Priority
5. **Lower learning rate** (Config files) - Improves convergence
6. **Fix sampling function clamping** (Line 45) - Ensures consistency
7. **Add gradient monitoring** (Already implemented) - For debugging

## Expected Outcomes After Fixes

1. **Stable Training**: Loss should decrease consistently without spikes
2. **Proper Convergence**: Models should learn effectively beyond first 100 steps  
3. **Numerical Stability**: No more overflow/underflow issues
4. **Consistent Behavior**: Training and inference should be numerically consistent

## Testing Strategy

1. **Start with small model**: Test fixes with minimal configuration first
2. **Monitor gradient norms**: Watch for explosion/vanishing
3. **Compare loss curves**: Should see smooth decrease without spikes
4. **Validate sampling**: Generated samples should be reasonable
5. **Scale up gradually**: Increase model size once basic functionality works

## Implementation Notes

- All fixes should be backward compatible
- Test with both mono and stereo configurations
- Verify fixes work with different bit depths (8-bit, 16-bit)
- Monitor memory usage during fixes
- Keep original categorical implementation intact
