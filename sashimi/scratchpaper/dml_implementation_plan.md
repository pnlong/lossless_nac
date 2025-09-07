# SaShiMi DML Implementation Plan

## Overview
This document outlines the changes needed to replace SaShiMi's categorical logits output with a Discretized Logistic Mixture (DML) output head while maintaining backwards compatibility with the original categorical output implementation.

## Files to Modify

### 1. `/src/models/sequence/modules/output_heads.py` (New File)
Create module containing both output head implementations:
```python
class CategoricalHead(nn.Module):
    """Original SaShiMi categorical output head"""
    def __init__(self, d_model, n_classes=256):
        super().__init__()
        self.projection = nn.Linear(d_model, n_classes)
        
    def forward(self, x):
        return self.projection(x)

class DiscretizedLogisticMixtureHead(nn.Module):
    """Optional DML output head"""
    def __init__(self, d_model, n_mixtures):
        super().__init__()
        self.n_mixtures = n_mixtures
        self.projection = nn.Linear(d_model, 3 * n_mixtures)
        
    def forward(self, x):
        h = self.projection(x)
        logit_probs, means, log_scales = torch.split(h, self.n_mixtures, dim=-1)
        mixture_weights = F.softmax(logit_probs, dim=-1)
        scales = F.softplus(log_scales)
        
        return {
            'mixture_weights': mixture_weights,
            'means': means,
            'scales': scales
        }

def get_output_head(head_type, d_model, **kwargs):
    """Factory function for output heads"""
    if head_type == 'categorical':
        return CategoricalHead(d_model, **kwargs)
    elif head_type == 'dml':
        return DiscretizedLogisticMixtureHead(d_model, **kwargs)
    else:
        raise ValueError(f"Unknown output head type: {head_type}")
```

### 2. `/src/models/sequence/backbones/sashimi.py`
Modify Sashimi backbone to use DML head:
```python
def __init__(self, d_model, output_head='categorical', n_mixtures=10, ...):
    # ... existing initialization ...
    self.output_head_type = output_head
    head_kwargs = {
        'n_mixtures': n_mixtures if output_head == 'dml' else None,
        'n_classes': self.dataset.n_tokens if output_head == 'categorical' else None
    }
    self.output_head = get_output_head(output_head, d_model, **head_kwargs)

def forward(self, x):
    # ... existing processing ...
    x = self.norm(x)
    return self.output_head(x), None
```

### 3. `/src/models/sequence/loss.py` (New File)
Add both loss implementations:
```python
def get_loss_fn(output_head_type):
    """Factory function for loss functions"""
    if output_head_type == 'categorical':
        return F.cross_entropy
    elif output_head_type == 'dml':
        return discretized_mix_logistic_loss
    else:
        raise ValueError(f"Unknown output head type: {output_head_type}")

def discretized_mix_logistic_loss(predictions, targets, dataset=None):
    """
    Compute negative log-likelihood for discretized mixture of logistics.
    
    Args:
        predictions: dict containing:
            - mixture_weights: [B, T, K] mixture probabilities (π)
            - means: [B, T, K] means (μ)
            - scales: [B, T, K] scales (s)
        targets: [B, T] quantized audio values in [0, num_classes-1]
        dataset: dataset object containing bits parameter
    """
    num_classes = 1 << dataset.bits  # 2^bits classes
    
    # Unpack predictions
    mixture_weights = predictions['mixture_weights']  # [B, T, K]
    means = predictions['means']                      # [B, T, K]
    scales = predictions['scales']                    # [B, T, K]
    
    # Rescale targets from [0, 2^bits-1] to [-1, 1]
    x = targets.float() / (num_classes/2) - 1        # [B, T]
    x = x.unsqueeze(-1)                             # [B, T, 1]
    
    # Calculate logistic CDFs
    def logistic_cdf(x, mean, scale):
        return torch.sigmoid((x - mean) / scale)
    
    # Get bin boundaries
    bin_size = 2.0 / num_classes
    x_plus = x + bin_size/2                         # Upper bin boundary
    x_minus = x - bin_size/2                        # Lower bin boundary
    
    # Handle edge cases for extreme bins
    left_edge = -1.0
    right_edge = 1.0
    
    # Compute CDFs at bin boundaries
    cdf_plus = logistic_cdf(x_plus, means, scales)   # [B, T, K]
    cdf_minus = logistic_cdf(x_minus, means, scales) # [B, T, K]
    
    # Calculate probability mass in bin as difference of CDFs
    bin_prob = cdf_plus - cdf_minus                  # [B, T, K]
    
    # Handle edge cases
    is_left_edge = (targets == 0)                    # [B, T]
    is_right_edge = (targets == num_classes-1)       # [B, T]
    
    # For leftmost bin, use CDF directly
    bin_prob = torch.where(
        is_left_edge.unsqueeze(-1),                  # [B, T, 1]
        cdf_plus,                                    # [B, T, K]
        bin_prob
    )
    
    # For rightmost bin, use complementary CDF
    bin_prob = torch.where(
        is_right_edge.unsqueeze(-1),                 # [B, T, 1]
        1 - cdf_minus,                               # [B, T, K]
        bin_prob
    )
    
    # Ensure numerical stability
    bin_prob = torch.clamp(bin_prob, min=1e-12)
    
    # Weight by mixture probabilities and sum
    log_probs = torch.log(bin_prob) + torch.log(mixture_weights)  # [B, T, K]
    log_probs = torch.logsumexp(log_probs, dim=-1)               # [B, T]
    
    # Average over batch and time dimensions
    nll = -log_probs.mean()
    
    # Add logging of mixture statistics
    with torch.no_grad():
        avg_scale = scales.mean().item()
        avg_mean = means.abs().mean().item()
        mixture_entropy = -(mixture_weights * torch.log(mixture_weights + 1e-12)).sum(-1).mean().item()
        
        return {
            'loss': nll,
            'avg_scale': avg_scale,
            'avg_mean': avg_mean,
            'mixture_entropy': mixture_entropy,
        }
```

### 4. `/src/dataloaders/audio.py`
Modify dataset classes to support both output types:
```python
class QuantizedAutoregressiveAudio(SequenceDataset):
    def __init__(self, output_head='categorical', n_mixtures=10, bits=8, ...):
        self.output_head = output_head
        self.n_mixtures = n_mixtures
        self.bits = bits
        # ... rest of init ...

    @property
    def d_output(self):
        if self.output_head == 'categorical':
            return 1 << self.bits  # Original behavior
        elif self.output_head == 'dml':
            return 3 * self.n_mixtures  # DML output size
```

### 5. `/configs/model/sashimi.yaml`
Add optional DML configuration:
```yaml
defaults:
  - layer: s4

_name_: sashimi
d_model: 64
n_layers: 8

# Output head configuration (new)
output_head: categorical  # ['categorical', 'dml']
n_mixtures: 10  # Only used if output_head='dml'

# Rest of original config
pool:
  - 4
  - 4
expand: 2
ff: 2
prenorm: True
dropout: 0.0
dropres: 0.0
initializer: null
transposed: True
residual: R
norm: layer
interp: 0
act_pool: null

layer:
  l_max: null
```

### 6. `/configs/task/audio_generation.yaml` (New File)
Create task config that adapts to output type:
```yaml
defaults:
  - loss: ${model.output_head}  # Automatically selects appropriate loss

_name_: audio_generation
metrics:
  - nll  # Log likelihood in bits per dimension
  - bpb  # Bits per byte (same as original)
```

### 7. `/configs/loss/categorical.yaml` (New File)
```yaml
_name_: cross_entropy
```

### 8. `/configs/loss/dml.yaml` (New File)
```yaml
_name_: dml
# num_classes will be inferred from dataset.bits
```

### 9. `/src/tasks/tasks.py` - **UPDATED: No new task needed**
Instead of creating a new AudioGenerationTask, we modified the BaseTask to handle DML losses by creating a closure that captures the dataset parameter:

```python
# Special handling for DML loss which needs dataset parameter
if hasattr(loss, '_name_') and loss._name_ == 'discretized_mix_logistic':
    # Create closure that captures dataset for DML loss
    dml_loss_fn = self.loss
    self.loss = lambda preds, targets, **kwargs: dml_loss_fn(preds, targets, dataset=self.dataset)
else:
    self.loss = U.discard_kwargs(self.loss)
```

This approach maintains backward compatibility while supporting DML without requiring a separate task class.

### 10. `/configs/experiment/audio/sashimi-dml.yaml` (New File)
Example experiment config using DML:
```yaml
# @package _global_
defaults:
  - /trainer: default
  - /loader: default
  - /dataset: musdb18mono
  - /task: audio_generation  # New task
  - /optimizer: adamw
  - /scheduler: plateau
  - /model: sashimi

model:
  output_head: dml
  n_mixtures: 10
  n_layers: 8
  dropout: 0.0

train:
  monitor: val/loss
  mode: min

task:
  metrics:
    - nll
    - bpb

loader:
  batch_size: 1

trainer:
  max_epochs: 1000

optimizer:
  lr: 0.004
```

## Training Script Updates

### 1. `/train.py`
No changes needed - the training script already handles dynamic loss functions through the task configuration.

### 2. Logging Updates
The metrics and losses are already handled through the task system.

## CLI Arguments
The output head type and number of mixtures can be set via command line using Hydra's override syntax:

```bash
# Train with default categorical output
python train.py experiment=audio/sashimi-musdb18mono

# Train with DML output
python train.py experiment=audio/sashimi-dml

# Override number of mixtures
python train.py experiment=audio/sashimi-dml model.n_mixtures=20
```

## Implementation Notes

1. **Default Values**
   - Default output_head: 'categorical' (maintains backwards compatibility)
   - Default n_mixtures: 10 (based on common usage in WaveNet/PixelCNN++)
   - num_classes: Automatically derived from dataset.bits (e.g., 256 for 8-bit audio)

2. **Config Hierarchy**
   - Dataset config defines bits parameter
   - Model config defines output head type and n_mixtures
   - Loss function automatically adapts to dataset's quantization

3. **Validation**
   - Add config validation to ensure n_mixtures is set when output_head='dml'
   - Verify dataset.bits is defined
   - Ensure bits parameter is consistent across dataset splits

## References

1. PixelCNN++ paper (original DML implementation)
2. WaveNet paper (audio application of DML)
3. SaShiMi paper (current architecture)

---

# SaShiMi DML Implementation Plan (Corrected & Extended)

## Overview
This document outlines the changes needed to extend SaShiMi’s categorical logits output with a **Discretized Logistic Mixture (DML)** output head, while maintaining backwards compatibility with the original categorical implementation.  

The corrected plan ensures:
- Numerically stable log-likelihood computation.  
- Proper handling of edge bins.  
- Clean integration into SaShiMi’s factory-based model/loss/task design.  
- Compatibility with existing configs.  

---

## Files to Modify

### 1. `/src/models/sequence/modules/output_heads.py` (New File)

```python
class CategoricalHead(nn.Module):
    """Original SaShiMi categorical output head"""
    def __init__(self, d_model, n_classes=256):
        super().__init__()
        self.projection = nn.Linear(d_model, n_classes)

    def forward(self, x):
        return self.projection(x)

class DiscretizedLogisticMixtureHead(nn.Module):
    """DML output head"""
    def __init__(self, d_model, n_mixtures):
        super().__init__()
        self.n_mixtures = n_mixtures
        self.projection = nn.Linear(d_model, 3 * n_mixtures)

    def forward(self, x):
        h = self.projection(x)
        logit_probs, means, log_scales = torch.split(h, self.n_mixtures, dim=-1)
        return {
            "logit_probs": logit_probs,   # unnormalized, will use log_softmax in loss
            "means": means,
            "log_scales": log_scales,     # stored in log-space for stability
        }

def get_output_head(head_type, d_model, **kwargs):
    if head_type == "categorical":
        return CategoricalHead(d_model, **kwargs)
    elif head_type == "dml":
        return DiscretizedLogisticMixtureHead(d_model, **kwargs)
    else:
        raise ValueError(f"Unknown output head type: {head_type}")

```

### 2. `/src/models/sequence/backbones/sashimi.py`

```python
def __init__(self, d_model, output_head="categorical", n_mixtures=10, bits=None, ...):
    # ... existing initialization ...
    self.output_head_type = output_head
    self.n_mixtures = n_mixtures
    self.bits = bits if bits is not None else 8  # Default to 8 bits
    self.n_classes = 1 << self.bits  # Calculate from bits

    head_kwargs = {
        "n_mixtures": self.n_mixtures if output_head == "dml" else None,
        "n_classes": self.n_classes if output_head == "categorical" else None,
    }
    self.output_head = get_output_head(output_head, d_model, **{k: v for k, v in head_kwargs.items() if v is not None})

def forward(self, x):
    # ... backbone processing ...
    x = self.norm(x)
    return self.output_head(x), None

```

### 3. `/src/models/sequence/loss.py` (New File)

```python
def get_loss_fn(output_head_type):
    if output_head_type == "categorical":
        return lambda preds, targets, **kwargs: {"loss": F.cross_entropy(preds, targets)}
    elif output_head_type == "dml":
        return discretized_mix_logistic_loss
    else:
        raise ValueError(f"Unknown output head type: {output_head_type}")

def discretized_mix_logistic_loss(predictions, targets, dataset=None):
    """
    Negative log-likelihood for discretized mixture of logistics.
    Args:
        predictions: dict with {logit_probs, means, log_scales}
        targets: [B, T] int tensor in [0, num_classes-1]
        dataset: provides dataset.bits
    """
    num_classes = 1 << dataset.bits
    bin_size = 2.0 / num_classes

    # Unpack predictions
    logit_probs = predictions["logit_probs"]   # [B, T, K]
    means = predictions["means"]               # [B, T, K]
    log_scales = predictions["log_scales"]     # [B, T, K]

    # Normalize targets to [-1, 1]
    x = targets.float() / (num_classes / 2) - 1
    x = x.unsqueeze(-1)  # [B, T, 1]

    # Stable scales
    inv_scales = torch.exp(-torch.clamp(log_scales, min=-7.0))  # avoid extreme small/large scales

    # Bin boundaries
    x_plus = x + bin_size / 2
    x_minus = x - bin_size / 2

    # Logistic CDF
    cdf_plus = torch.sigmoid((x_plus - means) * inv_scales)
    cdf_minus = torch.sigmoid((x_minus - means) * inv_scales)
    bin_prob = cdf_plus - cdf_minus

    # Handle edge cases
    is_left_edge = (targets == 0).unsqueeze(-1)
    is_right_edge = (targets == num_classes - 1).unsqueeze(-1)
    bin_prob = torch.where(is_left_edge, cdf_plus, bin_prob)
    bin_prob = torch.where(is_right_edge, 1 - cdf_minus, bin_prob)

    # Clamp for numerical stability
    bin_prob = torch.clamp(bin_prob, min=1e-12)

    # Mixture weighting
    log_probs = F.log_softmax(logit_probs, dim=-1) + torch.log(bin_prob)
    log_probs = torch.logsumexp(log_probs, dim=-1)  # [B, T]

    nll = -log_probs.mean()

    with torch.no_grad():
        avg_scale = log_scales.exp().mean().item()
        avg_mean = means.abs().mean().item()
        mixture_entropy = -(F.softmax(logit_probs, -1) * F.log_softmax(logit_probs, -1)).sum(-1).mean().item()

    return {
        "loss": nll,
        "avg_scale": avg_scale,
        "avg_mean": avg_mean,
        "mixture_entropy": mixture_entropy,
    }
```

### 4. `/src/dataloaders/audio.py`

```python
class QuantizedAutoregressiveAudio(SequenceDataset):
    def __init__(self, output_head="categorical", n_mixtures=10, bits=8, ...):
        self.output_head = output_head
        self.n_mixtures = n_mixtures
        self.bits = bits
        # ... rest of init ...

    @property
    def d_output(self):
        if self.output_head == "categorical":
            return 1 << self.bits
        elif self.output_head == "dml":
            return 3 * self.n_mixtures
```

### 5. `/configs/model/sashimi.yaml`

```yaml
_name_: sashimi
d_model: 64
n_layers: 8

output_head: categorical  # ['categorical', 'dml']
n_mixtures: 10            # only used if output_head='dml'

pool: [4, 4]
expand: 2
ff: 2
prenorm: True
dropout: 0.0
dropres: 0.0
initializer: null
transposed: True
residual: R
norm: layer
interp: 0
act_pool: null

layer:
  l_max: null
```

6. Loss Configs

- `/configs/loss/categorical.yaml`

```yaml
_name_: cross_entropy
```

- `/configs/loss/dml.yaml`

```yaml
_name_: dml
```

7. `/src/tasks/tasks.py`

```python
class AudioGenerationTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, predictions, targets):
        return self.loss(predictions, targets, dataset=self.dataset)
```

8. Example Experiment Config: `/configs/experiment/audio/sashimi-dml.yaml`

```yaml
# @package _global_
defaults:
  - /trainer: default
  - /loader: default
  - /dataset: musdb18mono
  - /task: audio_generation
  - /optimizer: adamw
  - /scheduler: plateau
  - /model: sashimi

model:
  output_head: dml
  n_mixtures: 10
  n_layers: 8
  dropout: 0.0

train:
  monitor: val/loss
  mode: min

task:
  metrics:
    - nll
    - bpb

loader:
  batch_size: 1

trainer:
  max_epochs: 1000

optimizer:
  lr: 0.004

```

## Training Script & CLI
No changes required in `train.py`. Use Hydra overrides:

```bash
# Default categorical
python train.py experiment=audio/sashimi-musdb18mono

# With DML
python train.py experiment=audio/sashimi-dml

# Override mixtures
python train.py experiment=audio/sashimi-dml model.n_mixtures=20
```

## Implementation Notes

- Default values: categorical head, 10 mixtures if DML.
- Numerical stability: log-scales clamped, bin probs clamped at `1e-12`.
- Backward compatibility: all configs still work if `output_head=categorical`.
- Validation: ensure `dataset.bits` is defined and consistent.

## References

- PixelCNN++ (Salimans et al., 2017)
- WaveNet (van den Oord et al., 2016)
- SaShiMi (Goel et al., 2022)

