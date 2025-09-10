"""Output heads for SaShiMi model.

This module provides different output heads that can be used with the SaShiMi backbone:
- CategoricalHead: Original categorical output (default)
- DiscretizedLogisticMixtureHead: DML output for more efficient high-bit audio
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CategoricalHead(nn.Module):
    """Original SaShiMi categorical output head"""
    def __init__(self, d_model, n_classes):
        super().__init__()
        self.projection = nn.Linear(d_model, n_classes)
        
    def forward(self, x):
        """
        Args:
            x: [..., d_model] backbone features
        Returns:
            [..., n_classes] logits for categorical distribution
        """
        return self.projection(x)

class DiscretizedLogisticMixtureHead(nn.Module):
    """DML output head that predicts parameters for a mixture of logistics.
    
    For each timestep, outputs parameters for K mixture components:
    - π: mixture weights (logits)
    - μ: component means
    - s: component scales (in log space for stability)
    
    The output dimension is 3K where K is n_mixtures.
    """
    def __init__(self, d_model, n_mixtures):
        super().__init__()
        self.n_mixtures = n_mixtures
        # Project to 3K outputs for K mixture components
        self.projection = nn.Linear(d_model, 3 * n_mixtures)
        
        # Better initialization for DML parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize DML parameters for better training stability."""
        with torch.no_grad():
            # Initialize mixture weights (logit_probs) to be roughly uniform
            # Small random values to break symmetry
            self.projection.weight[:self.n_mixtures].normal_(0, 0.1)
            self.projection.bias[:self.n_mixtures].fill_(0.0)
            
            # Initialize means to small values around zero
            self.projection.weight[self.n_mixtures:2*self.n_mixtures].normal_(0, 0.1)
            self.projection.bias[self.n_mixtures:2*self.n_mixtures].fill_(0.0)
            
            # Initialize log_scales to reasonable values (-1.0 to -0.5)
            # This gives scales around 0.37 to 0.61, which are good starting points
            self.projection.weight[2*self.n_mixtures:3*self.n_mixtures].normal_(0, 0.1)
            self.projection.bias[2*self.n_mixtures:3*self.n_mixtures].uniform_(-1.0, -0.5)
        
    def forward(self, x):
        """
        Args:
            x: [..., d_model] backbone features
        Returns:
            dict containing:
                logit_probs: [..., K] unnormalized mixture weights
                means: [..., K] component means
                log_scales: [..., K] log-scales for numerical stability
        """
        h = self.projection(x)  # [..., 3K]
        
        # Split into mixture parameters
        chunks = torch.chunk(h, 3, dim=-1)  # 3 tensors of shape [..., K]
        logit_probs, means, log_scales = chunks
        
        # No activation on logits - will use log_softmax in loss
        # No activation on means - can be any real value
        # No activation on log_scales - exponential in loss for positivity
        
        return {
            "logit_probs": logit_probs,  # Will apply softmax in loss
            "means": means,              # Raw values in (-inf, inf)
            "log_scales": log_scales,    # Log-space for numerical stability
        }

def get_output_head(head_type, d_model, **kwargs):
    """Factory function for output heads.
    
    Args:
        head_type: 'categorical' or 'dml'
        d_model: backbone feature dimension
        **kwargs: passed to specific head
            - n_classes for categorical
            - n_mixtures for dml
    """
    if head_type == "categorical":
        if "n_classes" not in kwargs:
            raise ValueError("n_classes required for categorical head")
        return CategoricalHead(d_model, kwargs["n_classes"])
    elif head_type == "dml":
        if "n_mixtures" not in kwargs:
            raise ValueError("n_mixtures required for DML head")
        return DiscretizedLogisticMixtureHead(d_model, kwargs["n_mixtures"])
    else:
        raise ValueError(f"Unknown output head type: {head_type}")