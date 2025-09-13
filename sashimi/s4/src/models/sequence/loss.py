"""Loss functions for SaShiMi model.

This module provides loss functions for different output heads:
- cross_entropy: For categorical output (default)
- discretized_mix_logistic_loss: For DML output
"""

import torch
import torch.nn.functional as F
import math
import numpy as np

def log_prob_from_logits(x):
    """Numerically stable log_softmax implementation that prevents overflow.

    Equivalent to TensorFlow's log_prob_from_logits function.
    """
    axis = -1  # Last dimension for mixture components
    m = torch.max(x, dim=axis, keepdim=True)[0]
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))

def sample_from_discretized_mix_logistic(predictions, num_classes, temperature=1.0):
    """Sample from discretized mixture of logistics distribution.

    PyTorch equivalent of TensorFlow's sample_from_discretized_mix_logistic.
    Adapted for 1D sequences (no channel dependencies like original's 3D images).

    Args:
        predictions: dict with {logit_probs, means, log_scales}
            - logit_probs: [..., K] unnormalized mixture weights
            - means: [..., K] component means
            - log_scales: [..., K] log-scales for numerical stability
        num_classes: number of quantization levels (2^bits)
        temperature: temperature for sampling (lower = more deterministic)

    Returns:
        samples: [...] sampled values in [0, num_classes-1]
    """
    # Unpack predictions
    logit_probs = predictions["logit_probs"]   # [..., K]
    means = predictions["means"]               # [..., K]
    log_scales = predictions["log_scales"]     # [..., K]

    # Clamp log_scales for stability (same as in loss function)
    log_scales = torch.clamp(log_scales, min=-5.0, max=5.0)

    batch_shape = logit_probs.shape[:-1]  # Shape without mixture dimension
    nr_mix = logit_probs.shape[-1]        # Number of mixture components

    # Sample mixture indicator using Gumbel-max trick
    # Equivalent to: tf.argmax(logit_probs - tf.log(-tf.log(tf.random_uniform(...))), 3)
    gumbels = -torch.log(-torch.log(torch.rand_like(logit_probs) * 0.999 + 1e-5))
    perturbed_logits = logit_probs / temperature + gumbels
    mixture_indices = torch.argmax(perturbed_logits, dim=-1, keepdim=True)

    # One-hot encode the selected mixture
    sel = torch.zeros_like(logit_probs).scatter_(-1, mixture_indices, 1.0)
    sel = sel.unsqueeze(-1)  # Add dimension for broadcasting with parameters

    # Select logistic parameters for the sampled mixture
    # Equivalent to: tf.reduce_sum(l[:,:,:,:,:nr_mix]*sel,4)
    selected_means = torch.sum(means.unsqueeze(-1) * sel, dim=-2).squeeze(-1)
    selected_log_scales = torch.sum(log_scales.unsqueeze(-1) * sel, dim=-2).squeeze(-1)

    # Sample from logistic distribution using inverse CDF
    # u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    # x = means + tf.exp(log_scales)*(tf.log(u) - tf.log(1. - u))
    u = torch.rand_like(selected_means) * 0.999 + 1e-5
    logistic_samples = selected_means + torch.exp(selected_log_scales) * (torch.log(u) - torch.log(1. - u))

    # Clip to [-1, 1] range (same as original)
    # x = tf.minimum(tf.maximum(x, -1.), 1.)
    logistic_samples = torch.clamp(logistic_samples, min=-1.0, max=1.0)

    # Convert from [-1, 1] back to [0, num_classes-1] (inverse of normalization in loss)
    # Equivalent to: (x + 1) * (num_classes / 2)
    samples = ((logistic_samples + 1.0) * (num_classes / 2)).long()

    # Ensure samples are within valid range
    samples = torch.clamp(samples, min=0, max=num_classes-1)

    return samples

def get_loss_fn(output_head_type):
    """Factory function for loss functions.
    
    Args:
        output_head_type: 'categorical' or 'dml'
    Returns:
        Loss function that takes (predictions, targets, dataset) and returns dict with 'loss' key
    """
    if output_head_type == "categorical":
        return lambda preds, targets, **kwargs: {"loss": F.cross_entropy(preds, targets)}
    elif output_head_type == "dml":
        return discretized_mix_logistic_loss
    else:
        raise ValueError(f"Unknown output head type: {output_head_type}")

def discretized_mix_logistic_loss(predictions, targets, dataset=None):
    """Negative log-likelihood for discretized mixture of logistics.

    Based on the original TensorFlow implementation from PixelCNN++.
    Assumes the data has been rescaled to [-1,1] interval.

    Args:
        predictions: dict with {logit_probs, means, log_scales}
            - logit_probs: [..., K] unnormalized mixture weights
            - means: [..., K] component means
            - log_scales: [..., K] log-scales for numerical stability
        targets: [...] integer tensor in [0, num_classes-1]
        dataset: provides dataset.bits for quantization levels

    Returns:
        dict containing:
            - loss: negative log likelihood
            - avg_scale: mean component scale
            - avg_mean: mean absolute component mean
            - mixture_entropy: entropy of mixture weights
    """
    if dataset is None:
        raise ValueError("dataset required for DML loss (need bits parameter)")

    if not hasattr(dataset, 'bits'):
        raise ValueError(f"dataset does not have bits attribute. Dataset type: {type(dataset)}")

    bits = dataset.bits
    if bits is None:
        raise ValueError("dataset.bits is None")

    num_classes = 1 << bits  # 2^bits classes

    # Unpack predictions - equivalent to original's unpacking
    logit_probs = predictions["logit_probs"]   # [..., K]
    means = predictions["means"]               # [..., K]
    log_scales = predictions["log_scales"]     # [..., K]

    # Handle stereo audio: targets and predictions should have matching sequence lengths
    # For stereo, both are doubled due to interleaving: (batch, seq_len*2) for targets, (batch, seq_len*2, K) for predictions
    # For mono, both are normal: (batch, seq_len) for targets, (batch, seq_len, K) for predictions
    
    # Get target sequence length
    if targets.dim() == 2:  # (batch, seq_len) or (batch, seq_len*2) for stereo
        target_seq_len = targets.shape[1]
    else:  # (batch, seq_len, 1) or similar
        target_seq_len = targets.shape[1]
    
    # Check if predictions have stereo-reshaped output (4D tensor) or regular output (3D tensor)
    if logit_probs.dim() == 4:  # Stereo reshaped: (batch, seq_len, 2, K)
        # Flatten stereo channels back to interleaved format for loss computation
        logit_probs = logit_probs.view(logit_probs.shape[0], -1, logit_probs.shape[-1])  # (batch, seq_len*2, K)
        means = means.view(means.shape[0], -1, means.shape[-1])  # (batch, seq_len*2, K)
        log_scales = log_scales.view(log_scales.shape[0], -1, log_scales.shape[-1])  # (batch, seq_len*2, K)
    
    # Ensure all prediction tensors have the same sequence length
    pred_seq_len = logit_probs.shape[1]
    assert logit_probs.shape[1] == means.shape[1] == log_scales.shape[1], \
        f"Mismatch in prediction sequence lengths: logit_probs={logit_probs.shape[1]}, means={means.shape[1]}, log_scales={log_scales.shape[1]}"
    
    # Ensure targets and predictions have matching sequence lengths
    if target_seq_len != pred_seq_len:
        raise ValueError(f"Target sequence length ({target_seq_len}) doesn't match prediction sequence length ({pred_seq_len}). "
                         f"This might indicate a stereo/mono mismatch in the model or dataset configuration.")

    # Normalize targets from [0, 2^bits-1] to [-1, 1] - equivalent to original's data preprocessing
    x = targets.float() / (num_classes / 2) - 1
    # Ensure x has the same number of dimensions as means for proper broadcasting
    if x.dim() == 3:  # targets was (batch, seq_len, 1)
        x = x.squeeze(-1)  # Remove the extra dimension to get (batch, seq_len)
    x = x.unsqueeze(-1)  # Add mixture dimension to get (batch, seq_len, 1)
    # Expand x to match the number of mixture components
    x = x.expand_as(means)  # Now x has shape (batch, seq_len, K) to match means

    # Clamp log_scales for stability - use symmetric clamping to prevent gradient explosion
    # Original uses tf.maximum(log_scales, -7.) but we need both min and max bounds
    log_scales = torch.clamp(log_scales, min=-5.0, max=5.0)

    # Compute inverse scales - equivalent to original's inv_stdv = tf.exp(-log_scales)
    inv_scales = torch.exp(-log_scales)
    
    # Compute bin boundaries - matches original's 1./255. discretization
    # For general bit depth: bin size should be 1.0 / (2^bits - 1)
    bin_size = 1.0 / (num_classes - 1)  # Original uses 1/255 for 8-bit
    
    # Use in-place operations where possible to save memory
    x_plus = x + bin_size / 2   # Upper boundary
    x_minus = x - bin_size / 2  # Lower boundary
    
    # Compute centered values for CDF calculations
    centered_x = x - means

    # Compute logistic CDF inputs - matches original's plus_in, min_in
    plus_in = inv_scales * (centered_x + bin_size / 2)  # equivalent to inv_stdv * (centered_x + 1./255.)
    min_in = inv_scales * (centered_x - bin_size / 2)   # equivalent to inv_stdv * (centered_x - 1./255.)

    # Compute CDFs - matches original
    cdf_plus = torch.sigmoid(plus_in)
    cdf_min = torch.sigmoid(min_in)
    cdf_delta = cdf_plus - cdf_min

    # Log-space probability computations - matches original's sophisticated approach
    # log_cdf_plus = plus_in - softplus(plus_in)  # log prob for edge case of 0
    log_cdf_plus = plus_in - F.softplus(plus_in)

    # log_one_minus_cdf_min = -softplus(min_in)  # log prob for edge case of 255
    log_one_minus_cdf_min = -F.softplus(min_in)

    # log_pdf_mid = mid_in - log_scales - 2.*softplus(mid_in)  # log prob in center of bin
    mid_in = inv_scales * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # Robust edge case selection - use proper quantization boundaries
    # Original uses: tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min,
    #                tf.where(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))

    # For our discretization, the equivalent of 127.5 is (num_classes-1)/2
    log_normalizer = math.log((num_classes - 1) / 2)

    # Use proper quantization-aware edge detection instead of arbitrary thresholds
    # Ensure targets has the right shape for edge detection
    if targets.dim() == 3:  # targets is (batch, seq_len, 1)
        targets_for_edge = targets.squeeze(-1)  # (batch, seq_len)
    else:
        targets_for_edge = targets  # (batch, seq_len)
    
    is_left_edge = (targets_for_edge == 0).unsqueeze(-1)  # [..., 1]
    is_right_edge = (targets_for_edge == num_classes - 1).unsqueeze(-1)  # [..., 1]
    
    # Expand edge detection to match mixture components
    is_left_edge = is_left_edge.expand_as(log_cdf_plus)  # [..., K]
    is_right_edge = is_right_edge.expand_as(log_cdf_plus)  # [..., K]
    
    # Select appropriate log probability based on quantization boundaries
    log_probs = torch.where(
        is_left_edge, 
        log_cdf_plus,
        torch.where(
            is_right_edge,
            log_one_minus_cdf_min,
            torch.where(
                cdf_delta > 1e-5,
                torch.log(torch.clamp(cdf_delta, min=1e-12)),
                log_pdf_mid - log_normalizer
            )
        )
    )

    # Add mixture weights - use PyTorch's optimized log_softmax for better efficiency and stability
    # Original uses tf.reduce_sum which is mathematically incorrect
    # Correct approach: add mixture weights to each component, then use logsumexp
    mixture_log_probs = F.log_softmax(logit_probs, dim=-1)  # (..., K) - mixture weights using PyTorch's optimized version
    log_probs = log_probs + mixture_log_probs  # (..., K) - add mixture weights to each component
    log_probs = torch.logsumexp(log_probs, dim=-1)  # (...,) - proper mixture computation

    # Negative log likelihood - matches original's -tf.reduce_sum(log_sum_exp(...))
    # Original has option for sum_all=True/False, we always use mean for consistency
    nll = -log_probs.mean()

    # Compute statistics for monitoring (unchanged)
    with torch.no_grad():
        avg_scale = log_scales.exp().mean().item()
        avg_mean = means.abs().mean().item()
        mixture_entropy = -(
            F.softmax(logit_probs, -1) * F.log_softmax(logit_probs, -1)
        ).sum(-1).mean().item()

    return {
        "loss": nll,
        "avg_scale": avg_scale,
        "avg_mean": avg_mean,
        "mixture_entropy": mixture_entropy,
    }