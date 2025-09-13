"""SaShiMi backbone with support for different output heads.

The backbone processes sequences through a U-Net style architecture with:
- Down blocks for downsampling
- Center blocks for processing
- Up blocks for upsampling
- Skip connections between corresponding layers

Supports both categorical and DML output heads.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.sequence.base import SequenceModule
from src.models.sequence.modules.pool import DownPool, UpPool
from src.models.sequence.backbones.block import SequenceResidualBlock
from src.models.sequence.modules.output_heads import get_output_head

class Sashimi(SequenceModule):
    def __init__(
        self,
        d_model,
        n_layers,
        pool=[],
        expand=1,
        ff=2,
        prenorm=False,
        dropout=0.0,
        dropres=0.0,
        layer=None,
        center_layer=None,
        residual=None,
        norm=None,
        initializer=None,
        transposed=True,
        interp=0,
        act_pool=None,
        d_input=None,              # New: actual input dimension (for stereo support)
        output_head='categorical',  # New: output head type
        n_mixtures=10,             # New: number of mixture components for DML
        bits=None,                 # New: number of bits for quantization (determines n_classes)
        is_stereo=False,           # New: whether processing stereo audio
        interleaving_strategy='temporal',  # New: interleaving strategy for stereo
        l_max=None,                # New: maximum sequence length for proper initialization
    ):
        super().__init__()

        self.d_model = d_model
        H = d_model
        # d_input is the raw input dimension, but after embedding it becomes d_model * channels
        # For pooling layers, we need the dimension after embedding
        self.d_input = d_input if d_input is not None else d_model

        self.interp = interp
        self.transposed = transposed

        # Output head configuration
        self.output_head_type = output_head
        self.n_mixtures = n_mixtures
        self.bits = bits if bits is not None else 8  # Default to 8 bits
        self.n_classes = 1 << self.bits  # Calculate n_classes from bits
        
        # Stereo configuration
        self.is_stereo = is_stereo
        self.interleaving_strategy = interleaving_strategy
        
        # Sequence length configuration
        self.l_max = l_max

        # Layer arguments
        layer_cfg = layer.copy()
        layer_cfg['dropout'] = dropout
        layer_cfg['transposed'] = self.transposed
        layer_cfg['initializer'] = initializer
        if self.l_max is not None:
            layer_cfg['l_max'] = self.l_max

        center_layer_cfg = center_layer if center_layer is not None else layer_cfg.copy()
        center_layer_cfg['dropout'] = dropout
        center_layer_cfg['transposed'] = self.transposed
        if self.l_max is not None:
            center_layer_cfg['l_max'] = self.l_max

        ff_cfg = {
            '_name_': 'ffn',
            'expand': ff,
            'transposed': self.transposed,
            'activation': 'gelu',
            'initializer': initializer,
            'dropout': dropout,
        }

        def _residual(d, i, layer):
            return SequenceResidualBlock(
                d,
                i,
                prenorm=prenorm,
                dropout=dropres,
                transposed=self.transposed,
                layer=layer,
                residual=residual if residual is not None else 'R',
                norm=norm,
                pool=None,
            )

        # Down blocks
        d_layers = []
        current_dim = self.d_model  # Start with model dimension (after embedding)
        for p in pool:
            # Add sequence downsampling and feature expanding
            d_layers.append(DownPool(current_dim, current_dim*expand, stride=p, transposed=self.transposed, activation=act_pool))
            current_dim *= expand
        self.d_layers = nn.ModuleList(d_layers)

        # Center block
        # Use current_dim (after pooling) instead of H (original d_model) for center block
        center_dim = current_dim  # This is the dimension after all pooling layers
        c_layers = [ ]
        for i in range(n_layers):
            c_layers.append(_residual(center_dim, i+1, center_layer_cfg))
            if ff > 0: c_layers.append(_residual(center_dim, i+1, ff_cfg))
        self.c_layers = nn.ModuleList(c_layers)

        # Up blocks
        u_layers = []
        for p in pool[::-1]:
            block = []
            current_dim //= expand  # Match the downsampling dimension progression
            block.append(UpPool(current_dim*expand, current_dim, stride=p, transposed=self.transposed, activation=act_pool))

            for i in range(n_layers):
                block.append(_residual(current_dim, i+1, layer_cfg))
                if ff > 0: block.append(_residual(current_dim, i+1, ff_cfg))

            u_layers.append(nn.ModuleList(block))

        self.u_layers = nn.ModuleList(u_layers)

        assert H == d_model

        self.norm = nn.LayerNorm(H)

        if interp > 0:
            interp_layers = []
            assert interp % 2 == 0
            for i in range(int(math.log2(interp))):
                block = []
                for j in range(2):
                    block.append(_residual(H, i+1, layer_cfg))
                    if ff > 0: block.append(_residual(H, i+1, ff_cfg))

                interp_layers.append(nn.ModuleList(block))

            self.interp_layers = nn.ModuleList(interp_layers)
            
        # Initialize output head
        head_kwargs = {
            "n_mixtures": self.n_mixtures if output_head == "dml" else None,
            "n_classes": self.n_classes if output_head == "categorical" else None,
        }
        self.output_head = get_output_head(
            output_head, 
            d_model, 
            **{k: v for k, v in head_kwargs.items() if v is not None}
        )

    @property
    def d_output(self):
        """Output dimension depends on output head type"""
        if self.output_head_type == "categorical":
            return self.n_classes
        elif self.output_head_type == "dml":
            return 3 * self.n_mixtures
        else:
            raise ValueError(f"Unknown output head type: {self.output_head_type}")
    
    def _deinterleave_stereo_tensor(self, tensor, batch_size, seq_len_interleaved, d_output):
        """
        Helper function to deinterleave a stereo tensor based on the interleaving strategy.
        
        Args:
            tensor: (batch_size, seq_len_interleaved, d_output) tensor
            batch_size: Batch size
            seq_len_interleaved: Length of interleaved sequence (should be seq_len * 2)
            d_output: Output dimension
            
        Returns:
            reshaped_tensor: (batch_size, seq_len, 2, d_output) tensor
        """
        seq_len = seq_len_interleaved // 2
        
        if self.interleaving_strategy == 'temporal':
            # Temporal interleaving: [L, R, L, R, ...] -> [L, L, L, ...], [R, R, R, ...]
            return tensor.view(batch_size, seq_len, 2, d_output)
        elif self.interleaving_strategy.startswith('blocking'):
            if self.interleaving_strategy == 'blocking':
                # Blocking interleaving: [L, L, L, ..., R, R, R, ...] -> [L, L, L, ...], [R, R, R, ...]
                return tensor.view(batch_size, 2, seq_len, d_output).transpose(1, 2)
            else: # some block size parameter is provided
                block_size = int(self.interleaving_strategy.split('-')[-1])
                if block_size <= 0: # Validate block size
                    raise ValueError(f"Block size must be positive, got {block_size}")
                block_size = min(block_size, seq_len) # Truncate block size if it's larger than sequence length                
                complete_blocks = seq_len // block_size # Calculate actual number of complete blocks and remainder
                remainder = seq_len % block_size # Calculate actual number of complete blocks and remainder
                if remainder == 0: # All blocks are complete
                    reshaped_tensor = tensor.view(batch_size, complete_blocks, 2, block_size, d_output)
                    return reshaped_tensor.view(batch_size, -1, 2, d_output)
                else: # Handle variable block sizes
                    complete_tensor = tensor[:, :complete_blocks * 2 * block_size, :]
                    reshaped_complete = complete_tensor.view(batch_size, complete_blocks, 2, block_size, d_output)
                    reshaped_complete = reshaped_complete.view(batch_size, -1, 2, d_output)
                    if remainder > 0: # Reshape final incomplete block
                        remainder_tensor = tensor[:, complete_blocks * 2 * block_size:, :]
                        reshaped_remainder = remainder_tensor.view(batch_size, 2, remainder, d_output).transpose(1, 2)
                        return torch.cat([reshaped_complete, reshaped_remainder], dim=1)
                    else:
                        return reshaped_complete
        else:
            raise ValueError(f"Unknown interleaving strategy: {self.interleaving_strategy}")

    def reshape_stereo_output(self, x):
        """
        Reshape stereo output from interleaved format back to stereo channels.
        
        Args:
            x: Either:
               - (batch, seq_len*2, d_output) tensor for categorical output
               - dict with keys ['logit_probs', 'means', 'log_scales'] for DML output
            
        Returns:
            x: Either:
               - (batch, seq_len, 2, d_output) tensor for categorical output
               - dict with reshaped tensors for DML output
        """
        if not self.is_stereo:
            return x
            
        # Handle DML dictionary output
        if isinstance(x, dict):
            reshaped_dict = {}
            for key, tensor in x.items():
                batch_size, seq_len_interleaved, d_output = tensor.shape
                reshaped_tensor = self._deinterleave_stereo_tensor(tensor, batch_size, seq_len_interleaved, d_output)
                reshaped_dict[key] = reshaped_tensor
            return reshaped_dict
        
        # Handle categorical tensor output
        batch_size, seq_len_interleaved, d_output = x.shape
        return self._deinterleave_stereo_tensor(x, batch_size, seq_len_interleaved, d_output)

    def forward(self, x, state=None, **kwargs):
        """
        input: (batch, length, d_model) - d_model may be scaled for multi-channel inputs
        output: (batch, length, d_output)
        """
        if self.interp > 0:
            # Interpolation will be used to reconstruct "missing" frames
            # Subsample the input sequence and run the SNet on that
            x_all = x
            x = x[:, ::self.interp, :]

            y = torch.zeros_like(x_all)
            # Run the interpolating layers
            interp_level = self.interp
            for block in self.interp_layers:
                # Pad to the right and discard the output of the first input
                # (creates dependence on the next time step for interpolation)
                z = x_all[:, ::interp_level, :]
                if self.transposed: z = z.transpose(1, 2)
                for layer in block:
                    z, _ = layer(z)

                z = F.pad(z[:, :, 1:], (0, 1), mode='replicate')
                if self.transposed: z = z.transpose(1, 2)
                y[:, interp_level//2 - 1::interp_level, :] += z
                interp_level = int(interp_level // 2)

        if self.transposed: x = x.transpose(1, 2)

        # print(f"🔍 DEBUG: SaShiMi forward - input shape: {x.shape}")
        # print(f"🔍 DEBUG: SaShiMi forward - d_input: {self.d_input}, d_model: {self.d_model}")
        # print(f"🔍 DEBUG: SaShiMi forward - number of down layers: {len(self.d_layers)}")

        # Down blocks
        outputs = []
        outputs.append(x)
        for i, layer in enumerate(self.d_layers):
            # print(f"🔍 DEBUG: SaShiMi forward - before layer {i}: {x.shape}")
            # print(f"🔍 DEBUG: SaShiMi forward - layer {i} type: {type(layer).__name__}")
            x, _ = layer(x)
            # print(f"🔍 DEBUG: SaShiMi forward - after layer {i}: {x.shape}")
            outputs.append(x)

        # Center block
        # print(f"🔍 DEBUG: SaShiMi forward - before center block: {x.shape}")
        for i, layer in enumerate(self.c_layers):
            # print(f"🔍 DEBUG: SaShiMi forward - before center layer {i}: {x.shape}")
            x, _ = layer(x)
            # print(f"🔍 DEBUG: SaShiMi forward - after center layer {i}: {x.shape}")
        x = x + outputs.pop() # add a skip connection to the last output of the down block
        # print(f"🔍 DEBUG: SaShiMi forward - after center block: {x.shape}")

        for block in self.u_layers:
            for layer in block:
                x, _ = layer(x)
                if isinstance(layer, UpPool):
                    # Before modeling layer in the block
                    x = x + outputs.pop()
                    outputs.append(x)
            x = x + outputs.pop() # add a skip connection from the input of the modeling part of this up block

        # feature projection
        if self.transposed: x = x.transpose(1, 2) # (batch, length, expand)
        x = self.norm(x)

        if self.interp > 0:
            y[:, self.interp - 1::self.interp, :] = x
            x = y

        # Apply output head
        # print(f"🔍 DEBUG: SaShiMi before output head: {x.shape}")
        x = self.output_head(x)
        # print(f"🔍 DEBUG: SaShiMi after output head: {x.shape}")
        
        # Reshape stereo output if needed
        if self.is_stereo:
            x = self.reshape_stereo_output(x)
            # print(f"🔍 DEBUG: SaShiMi after stereo reshaping: {x.shape}")
        
        return x, None

    def default_state(self, *args, **kwargs):
        """ x: (batch) """
        layers = list(self.d_layers) + list(self.c_layers) + [layer for block in self.u_layers for layer in block]
        return [layer.default_state(*args, **kwargs) for layer in layers]

    def step(self, x, state, **kwargs):
        """
        input: (batch, d_input)
        output: (batch, d_output)
        """
        # States will be popped in reverse order for convenience
        state = state[::-1]

        # Down blocks
        outputs = [] # Store all layers for SaShiMi
        next_state = []
        for layer in self.d_layers:
            outputs.append(x)
            x, _next_state = layer.step(x, state=state.pop(), **kwargs)
            next_state.append(_next_state)
            if x is None: break

        # Center block
        if x is None:
            # Skip computations since we've downsized
            skipped = len(self.d_layers) - len(outputs)
            for _ in range(skipped + len(self.c_layers)):
                next_state.append(state.pop())
            for i in range(skipped):
                for _ in range(len(self.u_layers[i])):
                    next_state.append(state.pop())
            u_layers = list(self.u_layers)[skipped:]
        else:
            outputs.append(x)
            for layer in self.c_layers:
                x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                next_state.append(_next_state)
            x = x + outputs.pop()
            u_layers = self.u_layers

        for block in u_layers:
            for layer in block:
                x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                next_state.append(_next_state)
                if isinstance(layer, UpPool):
                    # Before modeling layer in the block
                    x = x + outputs.pop()
                    outputs.append(x)
            x = x + outputs.pop()

        # feature projection and output head
        x = self.norm(x)
        x = self.output_head(x)
        return x, next_state