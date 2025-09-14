import copy
import os
import random
import time
from functools import partial, wraps
from typing import Callable, List, Optional

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from tqdm.auto import tqdm

import src.models.nn.utils as U
import src.utils as utils
import src.utils.train
from src.dataloaders import SequenceDataset  # TODO make registry
from src.tasks import decoders, encoders, tasks
from src.utils import registry
from src.utils.optim.ema import build_ema_optimizer
from src.utils.optim_groups import add_optimizer_hooks

log = src.utils.train.get_logger(__name__)

# Turn on TensorFloat32 (speeds up large model training substantially)
import torch.backends
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# Lots of annoying hacks to get WandbLogger to continuously retry on failure
class DummyExperiment:
    """Dummy experiment."""

    def nop(self, *args, **kw):
        pass

    def __getattr__(self, _):
        return self.nop

    def __getitem__(self, idx) -> "DummyExperiment":
        # enables self.logger.experiment[0].add_image(...)
        return self

    def __setitem__(self, *args, **kwargs) -> None:
        pass


def rank_zero_experiment(fn: Callable) -> Callable:
    """Returns the real experiment on rank 0 and otherwise the DummyExperiment."""

    @wraps(fn)
    def experiment(self):
        @rank_zero_only
        def get_experiment():
            return fn(self)

        return get_experiment() or DummyExperiment()

    return experiment


class CustomWandbLogger(WandbLogger):

    def __init__(self, *args, **kwargs):
        """Modified logger that insists on a wandb.init() call and catches wandb's error if thrown."""

        super().__init__(*args, **kwargs)

    @property
    @rank_zero_experiment
    def experiment(self):
        r"""
        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.
        Example::
        .. code-block:: python
            self.logger.experiment.some_wandb_function()
        """
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                # wandb process already created in this instance
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                # attach to wandb process referenced
                self._experiment = wandb._attach(attach_id)
            else:
                # create new wandb process
                while True:
                    try:
                        self._experiment = wandb.init(**self._wandb_init)
                        break
                    except Exception as e:
                        print("wandb Exception:\n", e)
                        t = random.randint(30, 60)
                        print(f"Sleeping for {t} seconds")
                        time.sleep(t)

                # define default x-axis
                if getattr(self._experiment, "define_metric", None):
                    self._experiment.define_metric("trainer/global_step")
                    self._experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)

        return self._experiment


class SequenceLightningModule(pl.LightningModule):
    def __init__(self, config):
        # Disable profiling executor. This reduces memory and increases speed.
        try:
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
        except AttributeError:
            pass

        super().__init__()
        # Passing in config expands it one level, so can access by self.hparams.train instead of self.hparams.config.train
        self.save_hyperparameters(config, logger=False)

        # Dataset arguments - include chunk_size parameters from train section
        dataset_kwargs = dict(self.hparams.dataset)

        # Add chunk_size parameters from train section if they exist
        if hasattr(self.hparams.train, 'train_chunk_size'):
            dataset_kwargs['train_chunk_size'] = self.hparams.train.train_chunk_size
        if hasattr(self.hparams.train, 'val_chunk_size'):
            dataset_kwargs['val_chunk_size'] = self.hparams.train.val_chunk_size

        self.dataset = SequenceDataset.registry[self.hparams.dataset._name_](
            **dataset_kwargs
        )

        # Setup dataset immediately to avoid timing issues with sanity check
        if not getattr(self.hparams.train, 'disable_dataset', True):
            # print("Setting up dataset in __init__ for sanity check compatibility...")
            self.dataset.setup()
            # print("Dataset setup in __init__ complete")

        # Check hparams
        self._check_config()

        # PL has some bugs, so add hooks and make sure they're only called once
        self._has_setup = False

        self.setup()  ## Added by KS

    def setup(self, stage=None):
        # print(f"Model setup called with stage={stage}")
        # Setup dataset if not already done (for cases where __init__ didn't run setup)
        if not self.hparams.train.disable_dataset and (not hasattr(self.dataset, 'dataset_train') or self.dataset.dataset_train is None):
            # print("Setting up dataset in setup method...")
            self.dataset.setup()
            # print("Dataset setup in setup method complete")

        # We need to set up the model in setup() because for some reason when training with DDP, one GPU uses much more memory than the others
        # In order to not overwrite the model multiple times during different stages, we need this hack
        # TODO PL 1.5 seems to have an option to skip hooks to avoid this
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/5410#issuecomment-762257024
        if self._has_setup:
            return
        else:
            self._has_setup = True

        # Convenience feature: if model specifies encoder, combine it with main encoder
        encoder_cfg = utils.to_list(self.hparams.encoder) + utils.to_list(
            self.hparams.model.pop("encoder", None)
        )
        decoder_cfg = utils.to_list(
            self.hparams.model.pop("decoder", None)
        ) + utils.to_list(self.hparams.decoder)

        # Instantiate model
        model_config = self.hparams.model.copy()
        
        # Pass bits parameter from dataset config to model for output head initialization
        if 'bits' not in model_config and hasattr(self.dataset, 'bits'):
            model_config['bits'] = self.dataset.bits
            
        # Ensure d_input matches the actual input dimension (always 1 for both mono and stereo)
        raw_d_input = getattr(self.dataset, 'd_input', 1)
        model_config['d_input'] = raw_d_input
        # print(f"🔍 DEBUG: Setting d_input to {raw_d_input}")
        
        # Pass sequence length and stereo information to model for stereo
        if hasattr(self.dataset, 'is_stereo') and self.dataset.is_stereo:
            # For stereo, the model needs to know about the doubled sequence length
            if '__l_max' in self.hparams.dataset:
                model_config['l_max'] = self.hparams.dataset['__l_max']
                # print(f"🔍 DEBUG: Setting l_max to {model_config['l_max']} for stereo")
            
            # Pass stereo configuration to model
            model_config['is_stereo'] = True
            model_config['interleaving_strategy'] = getattr(self.dataset, 'interleaving_strategy', 'temporal')
            print(f"🔍 DEBUG: Setting stereo config: is_stereo=True, strategy={model_config['interleaving_strategy']}")
        else:
            # For mono, also pass l_max if available
            if '__l_max' in self.hparams.dataset:
                model_config['l_max'] = self.hparams.dataset['__l_max']
                # print(f"🔍 DEBUG: Setting l_max to {model_config['l_max']} for mono")
                
        # print(f"🔍 DEBUG: Model config before instantiation: {model_config}")
        print(f"🔍 DEBUG: Dataset d_input: {getattr(self.dataset, 'd_input', 'Not set')}")
        print(f"🔍 DEBUG: Dataset is_stereo: {getattr(self.dataset, 'is_stereo', 'Not set')}")
        print(f"🔍 DEBUG: Dataset interleaving_strategy: {getattr(self.dataset, 'interleaving_strategy', 'Not set')}")
        self.model = utils.instantiate(registry.model, model_config)
        # print("🔍 DEBUG: Model instantiated successfully!")
        # print(f"🔍 DEBUG: Model instantiated with d_model: {self.model.d_model}")
        # print(f"🔍 DEBUG: Model d_input: {getattr(self.model, 'd_input', 'Not set')}")
        if (name := self.hparams.train.post_init_hook['_name_']) is not None:
            kwargs = self.hparams.train.post_init_hook.copy()
            del kwargs['_name_']
            for module in self.modules():
                if hasattr(module, name):
                    getattr(module, name)(**kwargs)

        task_config = self.hparams.task.copy()
        if hasattr(self.hparams.model, 'output_head') and self.hparams.model.output_head == 'dml':
            task_config['loss'] = 'dml'
            print("🚀 " + "="*50)
            print("🚀 DML MODE ACTIVATED!")
            print("🚀 Using Discretized Mixture of Logistics output head")
            print("🚀 Model will output mixture parameters instead of class logits")
            print("🚀 Loss function: discretized_mix_logistic_loss")
            print("🚀 " + "="*50)

        # Debug: Check if S4 layers are properly instantiated
        # print("🔍 DEBUG: Checking model architecture...")
        # print(f"🔍 DEBUG: Model type: {type(self.model)}")
        # print(f"🔍 DEBUG: Model class name: {self.model.__class__.__name__}")

        # Check for S4/SSM layers (by both name and type)
        s4_layers_found = []
        ssm_layers_found = []
        total_layers = 0

        for name, module in self.model.named_modules():
            total_layers += 1
            # Check by module type (more reliable)
            module_class_name = module.__class__.__name__
            module_full_name = str(type(module))

            # S4-related modules
            if ('S4' in module_class_name or
                'S4Block' in module_class_name or
                'FFTConv' in module_class_name or
                'SSMKernel' in module_class_name):
                s4_layers_found.append((name, type(module)))

            # SSM-related modules
            if ('SSM' in module_class_name or
                'SSMKernel' in module_class_name):
                ssm_layers_found.append((name, type(module)))

        # print(f"🔍 DEBUG: Total model layers: {total_layers}")
        # print(f"🔍 DEBUG: S4 layers found: {len(s4_layers_found)}")
        # for name, layer_type in s4_layers_found[:5]:  # Show first 5
        #     print(f"🔍 DEBUG:   - {name}: {layer_type}")

        # print(f"🔍 DEBUG: SSM layers found: {len(ssm_layers_found)}")
        # for name, layer_type in ssm_layers_found[:5]:  # Show first 5
        #     print(f"🔍 DEBUG:   - {name}: {layer_type}")

        # Show layer hierarchy for first few c_layers
        # print("🔍 DEBUG: Inspecting layer hierarchy...")
        # for name, module in list(self.model.named_modules())[:20]:  # First 20 modules
        #     if 'c_layers' in name and name.count('.') <= 2:  # Top-level c_layers
        #         print(f"🔍 DEBUG:   {name}: {type(module)}")
        #         # Show children
        #         for child_name, child_module in module.named_modules():
        #             if child_name and '.' not in child_name:  # Direct children only
        #                 print(f"🔍 DEBUG:     └─ {child_name}: {type(child_module)}")

        # Check model configuration
        if hasattr(self.model, 'output_head_type'):
            print(f"🔍 DEBUG: Model output_head_type: {self.model.output_head_type}")
        if hasattr(self.model, 'n_mixtures'):
            print(f"🔍 DEBUG: Model n_mixtures: {self.model.n_mixtures}")
        if hasattr(self.model, 'bits'):
            print(f"🔍 DEBUG: Model bits: {self.model.bits}")
        if hasattr(self.model, 'n_classes'):
            print(f"🔍 DEBUG: Model n_classes: {self.model.n_classes}")
        if hasattr(self.model, 'd_output'):
            print(f"🔍 DEBUG: Model d_output: {self.model.d_output}")

        # Check if model has the expected backbone structure
        # if hasattr(self.model, 'backbone') or hasattr(self.model, 'layers'):
        #     backbone_attr = 'backbone' if hasattr(self.model, 'backbone') else 'layers'
        #     backbone = getattr(self.model, backbone_attr)
        #     print(f"🔍 DEBUG: Model {backbone_attr}: {type(backbone)}")
        #     if hasattr(backbone, '__len__'):
        #         print(f"🔍 DEBUG: Backbone length: {len(backbone)}")
        #         for i, layer in enumerate(backbone):
        #             print(f"🔍 DEBUG:   Layer {i}: {type(layer)}")
        #             if i >= 3:  # Only show first few layers
        #                 print(f"🔍 DEBUG:   ... and {len(backbone) - i - 1} more layers")
        #                 break

        # Check for kernel optimization modules
        kernel_modules_found = []
        for name, module in self.model.named_modules():
            module_str = str(type(module)).lower()
            if any(keyword in module_str for keyword in ['cauchy', 'vandermonde', 'fft', 'ssm', 's4']):
                kernel_modules_found.append((name, type(module)))

        # print(f"🔍 DEBUG: Kernel-related modules found: {len(kernel_modules_found)}")
        # for name, module_type in kernel_modules_found[:5]:  # Show first 5
        #     print(f"🔍 DEBUG:   - {name}: {module_type}")

        # Check if any modules have been imported that would trigger kernel warnings
        import sys
        kernel_related_imports = []
        for module_name, module in sys.modules.items():
            if module and any(keyword in module_name.lower() for keyword in ['cauchy', 'vandermonde', 'extensions', 'kernels']):
                kernel_related_imports.append(module_name)

        # print(f"🔍 DEBUG: Kernel-related imports: {len(kernel_related_imports)}")
        # for imp in kernel_related_imports[:5]:  # Show first 5
        #     print(f"🔍 DEBUG:   - {imp}")

        # print("🔍 DEBUG: Model architecture check complete")
        # print("="*60)

        self.task = utils.instantiate(
            tasks.registry, task_config, dataset=self.dataset, model=self.model
        )

        # Create encoders and decoders
        encoder = encoders.instantiate(
            encoder_cfg, model=self.model, dataset=self.dataset
        )
        decoder = decoders.instantiate(
            decoder_cfg, model=self.model, dataset=self.dataset
        )

        # Debug: Check if task already has an encoder (stereo case)
        # print(f"🔍 DEBUG: Task encoder type: {type(self.task.encoder).__name__}")
        # print(f"🔍 DEBUG: Config encoder type: {type(encoder).__name__}")
        # print(f"🔍 DEBUG: Encoder config: {encoder_cfg}")
        
        # Extract the modules so they show up in the top level parameter count
        # Use task encoder if available, otherwise use config encoder
        # print(f"🔍 DEBUG: Task type: {type(self.task).__name__}")
        # print(f"🔍 DEBUG: Task has encoder: {hasattr(self.task, 'encoder')}")
        # if hasattr(self.task, 'encoder'):
            # print(f"🔍 DEBUG: Task encoder is not None: {self.task.encoder is not None}")
            # print(f"🔍 DEBUG: Task encoder type: {type(self.task.encoder).__name__}")
        
        # Use task encoder if available, otherwise use config encoder
        if hasattr(self.task, 'encoder') and self.task.encoder is not None:
            # print(f"🔍 DEBUG: ✅ Using task encoder (type: {type(self.task.encoder).__name__})")
            self.encoder = self.task.encoder
        else:
            # print(f"🔍 DEBUG: ❌ Task encoder not available, using config encoder")
            # print(f"🔍 DEBUG: Config encoder type: {type(encoder).__name__}")
            self.encoder = encoder
            
        self.decoder = U.PassthroughSequential(decoder, self.task.decoder)
        self.loss = self.task.loss
        self.loss_val = self.task.loss
        if hasattr(self.task, 'loss_val'):
            self.loss_val = self.task.loss_val
        self.metrics = self.task.metrics

        # Handle state logic
        self._initialize_state()

        # Debug: Parameter statistics for output head
        print("🏋️  " + "="*50)
        print("🏋️  PARAMETER STATISTICS")
        
        # Calculate total model parameters (encoder + backbone + decoder)
        n_params = sum(p.numel() for p in self.parameters())
        
        # Calculate output head parameters
        # The output head is part of the backbone model
        n_params_output_head = 0
        if hasattr(self.model, 'output_head') and self.model.output_head is not None:
            n_params_output_head = sum(p.numel() for p in self.model.output_head.parameters())
        
        # Calculate percentage
        percentage_output_head = (100 * n_params_output_head / n_params) if n_params > 0 else 0
        
        print(f"🏋️  Number of Parameters in Output Head: {n_params_output_head:,}")
        print(f"🏋️  Total Number of Parameters in Model: {n_params:,}")
        print(f"🏋️  Percentage of Parameters in Output Head: {percentage_output_head:.2f}%")
        print("🏋️  " + "="*50)

    def load_state_dict(self, state_dict, strict=True):
        if self.hparams.train.pretrained_model_state_hook['_name_'] is not None:
            model_state_hook = utils.instantiate(
                registry.model_state_hook,
                self.hparams.train.pretrained_model_state_hook.copy(),
                partial=True,
            )
            # Modify the checkpoint['state_dict'] inside model_state_hook e.g. to inflate 2D convs to 3D convs
            state_dict = model_state_hook(self.model, state_dict)

        print("Custom load_state_dict function is running.")

        # note, it needs to return something from the normal function we overrided
        return super().load_state_dict(state_dict, strict=strict)

    def _check_config(self):
        assert self.hparams.train.state.mode in [None, "none", "null", "reset", "bptt", "tbptt"]
        assert (
            (n := self.hparams.train.state.n_context) is None
            or isinstance(n, int)
            and n >= 0
        )
        assert (
            (n := self.hparams.train.state.n_context_eval) is None
            or isinstance(n, int)
            and n >= 0
        )

    def _initialize_state(self):
        """Called at model setup and start of epoch to completely reset state"""
        self._state = None
        self._memory_chunks = []

    def _reset_state(self, batch, device=None):
        """Called to construct default_state when necessary, e.g. during BPTT"""
        device = device or batch[0].device
        self._state = self.model.default_state(*batch[0].shape[:1], device=device)

    def _detach_state(self, state):
        if isinstance(state, torch.Tensor):
            return state.detach()
        elif isinstance(state, tuple):
            return tuple(self._detach_state(s) for s in state)
        elif isinstance(state, list):
            return [self._detach_state(s) for s in state]
        elif isinstance(state, dict):
            return {k: self._detach_state(v) for k, v in state.items()}
        elif state is None:
            return None
        else:
            raise NotImplementedError

    def _process_state(self, batch, batch_idx, train=True):
        """Handle logic for state context."""
        # Number of context steps
        key = "n_context" if train else "n_context_eval"
        n_context = self.hparams.train.state.get(key)

        # Don't need to do anything if 0 context steps. Make sure there is no state
        if n_context == 0 and self.hparams.train.state.mode not in ['tbptt']:
            self._initialize_state()
            return

        # Reset state if needed
        if self.hparams.train.state.mode == "reset":
            if batch_idx % (n_context + 1) == 0:
                self._reset_state(batch)

        # Pass through memory chunks
        elif self.hparams.train.state.mode == "bptt":
            self._reset_state(batch)
            with torch.no_grad():  # should be unnecessary because individual modules should handle this
                for _batch in self._memory_chunks:
                    self.forward(_batch)
            # Prepare for next step
            self._memory_chunks.append(batch)
            self._memory_chunks = self._memory_chunks[-n_context:]

        elif self.hparams.train.state.mode == 'tbptt':
            _, _, z = batch
            reset = z["reset"]
            if reset:
                self._reset_state(batch)
            else:
                self._state = self._detach_state(self._state)

    def _on_epoch_start(self):
        self._initialize_state()

    def forward(self, batch):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        x, y, *z = batch # z holds extra dataloader info such as resolution
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]

        # print(f"🔍 DEBUG: Before encoder: {x.shape}")
        # print(f"🔍 DEBUG: Before encoder dtype: {x.dtype}")
        # print(f"🔍 DEBUG: Before encoder sample values: {x[0, :5, :5] if x.dim() == 3 else x[0, :5]}")
        
        # No special stereo handling needed - data is already interleaved
        # x shape: (batch, seq_len, 1) for mono or (batch, seq_len*2, 1) for stereo
        
        # print(f"🔍 DEBUG: Forward pass input shape: {x.shape}")
        # print(f"🔍 DEBUG: Forward pass input dtype: {x.dtype}")
        
        x, w = self.encoder(x, **z) # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
        # print(f"🔍 DEBUG: After encoder shape: {x.shape}")
        # print(f"🔍 DEBUG: After encoder dtype: {x.dtype}")
        # print(f"🔍 DEBUG: After encoder: {x.shape}")
        # print(f"🔍 DEBUG: After encoder dtype: {x.dtype}")
        # print(f"🔍 DEBUG: After encoder sample values: {x[0, :5, :5] if x.dim() == 3 else x[0, :5]}")
        x, state = self.model(x, **w, state=self._state)
        # print(f"🔍 DEBUG: After model: {x.shape}")
        self._state = state
        x, w = self.decoder(x, state=state, **z)
        # print(f"🔍 DEBUG: After decoder: {x.shape}")
        return x, y, w

    def step(self, x_t):
        x_t, *_ = self.encoder(x_t) # Potential edge case for encoders that expect (B, L, H)?
        x_t, state = self.model.step(x_t, state=self._state)
        self._state = state
        # x_t = x_t[:, None, ...] # Dummy length
        # x_t, *_ = self.decoder(x_t, state=state)
        # x_t = x_t[:, 0, ...]
        x_t, *_ = self.decoder.step(x_t, state=state)
        return x_t

    def _shared_step(self, batch, batch_idx, prefix="train"):

        self._process_state(batch, batch_idx, train=(prefix == "train"))

        # print(f"🔍 DEBUG: Original batch target shape: {batch[1].shape}")
        x, y, w = self.forward(batch)
        
        # print(f"🔍 DEBUG: Model output shape: {x.shape}")
        # print(f"🔍 DEBUG: Target shape: {y.shape}")
        # print(f"🔍 DEBUG: Target dtype: {y.dtype}")

        # Loss
        if prefix == 'train':
            loss = self.loss(x, y, **w)
        else:
            loss = self.loss_val(x, y, **w)

        # Metrics
        metrics = self.metrics(x, y, **w)

        # Handle DML statistics computation for monitoring
        if hasattr(self.model, 'output_head_type') and self.model.output_head_type == 'dml':
            from src.models.sequence.loss import compute_dml_statistics
            stats = compute_dml_statistics(x)
            # Add DML-specific metrics for monitoring
            metrics["avg_scale"] = stats["avg_scale"]
            metrics["avg_mean"] = stats["avg_mean"]
            metrics["mixture_entropy"] = stats["mixture_entropy"]

        metrics["loss"] = loss

        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Calculate torchmetrics: these are accumulated and logged at the end of epochs
        self.task.torchmetrics(x, y, prefix)

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_start(self):
        self._on_epoch_start()
        # Reset training torchmetrics
        self.task._reset_torchmetrics("train")


    def on_train_epoch_end(self):
        # Advance training chunk index in dataset for next epoch
        if hasattr(self.dataset, 'use_train_chunking') and self.dataset.use_train_chunking:
            if hasattr(self.dataset, '_current_train_chunk'):
                old_chunk = self.dataset._current_train_chunk
                total_chunks = (len(self.dataset.dataset_train) + self.dataset.train_chunk_size - 1) // self.dataset.train_chunk_size
                self.dataset._current_train_chunk = (self.dataset._current_train_chunk + 1) % total_chunks

        # Log training torchmetrics
        super().on_train_epoch_end()
        self.log_dict(
            {f"train/{k}": v for k, v in self.task.get_torchmetrics("train").items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )

    def on_validation_epoch_start(self):
        self._on_epoch_start()
        # Reset all validation torchmetrics
        for name in self.val_loader_names:
            self.task._reset_torchmetrics(name)


    def on_validation_epoch_end(self):
        # Log all validation torchmetrics FIRST
        super().on_validation_epoch_end()
        for name in self.val_loader_names:
            self.log_dict(
                {f"{name}/{k}": v for k, v in self.task.get_torchmetrics(name).items()},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
                sync_dist=True,
            )

        # Advance validation chunk index in dataset for next epoch (AFTER logging)
        if hasattr(self.dataset, 'use_val_chunking') and self.dataset.use_val_chunking:
            if hasattr(self.dataset, '_current_val_chunk'):
                old_chunk = self.dataset._current_val_chunk
                self.dataset._current_val_chunk = (self.dataset._current_val_chunk + 1) % (
                    (len(self.dataset.dataset_val) + self.dataset.val_chunk_size - 1) // self.dataset.val_chunk_size
                )

        # Also advance test chunk index if test dataloaders are included in validation
        # (This happens when test loaders are processed as part of validation phase)
        if hasattr(self.dataset, 'use_val_chunking') and self.dataset.use_val_chunking:
            if hasattr(self.dataset, '_current_test_chunk'):
                old_chunk = self.dataset._current_test_chunk
                self.dataset._current_test_chunk = (self.dataset._current_test_chunk + 1) % (
                    (len(self.dataset.dataset_test) + self.dataset.val_chunk_size - 1) // self.dataset.val_chunk_size
                )

    def on_test_epoch_start(self):
        self._on_epoch_start()
        # Reset all test torchmetrics
        for name in self.test_loader_names:
            self.task._reset_torchmetrics(name)

    def on_test_epoch_end(self):
        # Log all test torchmetrics
        super().on_test_epoch_end()
        for name in self.test_loader_names:
            self.log_dict(
                {f"{name}/{k}": v for k, v in self.task.get_torchmetrics(name).items()},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
                sync_dist=True,
            )

        # Advance test chunk index in dataset for next epoch (AFTER logging)
        if hasattr(self.dataset, 'use_val_chunking') and self.dataset.use_val_chunking:
            if hasattr(self.dataset, '_current_test_chunk'):
                old_chunk = self.dataset._current_test_chunk
                self.dataset._current_test_chunk = (self.dataset._current_test_chunk + 1) % (
                    (len(self.dataset.dataset_test) + self.dataset.val_chunk_size - 1) // self.dataset.val_chunk_size
                )

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, prefix="train")

        # Add gradient clipping for DML models to improve training stability
        if hasattr(self.model, 'output_head_type') and self.model.output_head_type == 'dml':
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # Log gradient norm for monitoring
            total_norm = 0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            # Log gradient norm
            self.log("trainer/grad_norm", total_norm, on_step=True, on_epoch=False, 
                    prog_bar=False, add_dataloader_idx=False, sync_dist=True)

        # Log the loss explicitly so it shows up in WandB
        # Note that this currently runs into a bug in the progress bar with ddp (as of 1.4.6)
        # https://github.com/PyTorchLightning/pytorch-lightning/pull/9142
        # We additionally log the epochs under 'trainer' to get a consistent prefix with 'global_step'
        loss_epoch = {"trainer/loss": loss, "trainer/epoch": self.current_epoch}
        self.log_dict(
            loss_epoch,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        # Log any extra info that the models want to expose (e.g. output norms)
        metrics = {}
        for module in list(self.modules())[1:]:
            if hasattr(module, "metrics"):
                metrics.update(module.metrics)

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Check if this is an EMA validation dataloader and if EMA is enabled
        is_ema_dataloader = self.val_loader_names[dataloader_idx].endswith("/ema")
        optimizer_has_stepped = hasattr(self.optimizers().optimizer, 'stepped')

        # EMA validation only happens if:
        # 1. It's an EMA dataloader (name ends with "/ema")
        # 2. EMA is enabled (optimizer has stepped attribute)
        # 3. The optimizer has actually stepped (to avoid first epoch issues)
        ema = (
            is_ema_dataloader
            and optimizer_has_stepped
            and self.optimizers().optimizer.stepped
        )


        if ema:
            self.optimizers().swap_ema()
        loss = self._shared_step(
            batch, batch_idx, prefix=self.val_loader_names[dataloader_idx]
        )
        if ema:
            self.optimizers().swap_ema()

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(
            batch, batch_idx, prefix=self.test_loader_names[dataloader_idx]
        )

    def configure_optimizers(self):

        # Set zero weight decay for some params
        if 'optimizer_param_grouping' in self.hparams.train:
            add_optimizer_hooks(self.model, **self.hparams.train.optimizer_param_grouping)

        # Normal parameters
        all_params = list(self.parameters())
        params = [p for p in all_params if not hasattr(p, "_optim")]


        # Construct optimizer, add EMA if necessary
        if self.hparams.train.ema > 0.0:
            optimizer = utils.instantiate(
                registry.optimizer,
                self.hparams.optimizer,
                params,
                wrap=build_ema_optimizer,
                polyak=self.hparams.train.ema,
            )
        else:
            optimizer = utils.instantiate(registry.optimizer, self.hparams.optimizer, params)

        del self.hparams.optimizer._name_

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
        hps = [
            # dict(s) for s in set(frozenset(hp.items()) for hp in hps)
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
            # dict(s) for s in dict.fromkeys(frozenset(hp.items()) for hp in hps)
        ]  # Unique dicts
        print("Hyperparameter groups", hps)
        for hp in hps:
            params = [p for p in all_params if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **self.hparams.optimizer, **hp}
            )

        ### Layer Decay ###

        if self.hparams.train.layer_decay['_name_'] is not None:
            get_num_layer = utils.instantiate(
                registry.layer_decay,
                self.hparams.train.layer_decay['_name_'],
                partial=True,
            )

            # Go through all parameters and get num layer
            layer_wise_groups = {}
            num_max_layers = 0
            for name, p in self.named_parameters():
                # Get layer id for each parameter in the model
                layer_id = get_num_layer(name)

                # Add to layer wise group
                if layer_id not in layer_wise_groups:
                    layer_wise_groups[layer_id] = {
                        'params': [],
                        'lr': None,
                        'weight_decay': self.hparams.optimizer.weight_decay
                    }
                layer_wise_groups[layer_id]['params'].append(p)

                if layer_id > num_max_layers: num_max_layers = layer_id

            # Update lr for each layer
            for layer_id, group in layer_wise_groups.items():
                group['lr'] = self.hparams.optimizer.lr * (self.hparams.train.layer_decay.decay ** (num_max_layers - layer_id))

            # Reset the torch optimizer's param groups
            optimizer.param_groups = []
            for layer_id, group in layer_wise_groups.items():
                optimizer.add_param_group(group)

        # Print optimizer info for debugging
        keys = set([k for hp in hps for k in hp.keys()])  # Special hparams
        utils.train.log_optimizer(log, optimizer, keys)

        # Configure scheduler
        if "scheduler" not in self.hparams:
            return optimizer
        lr_scheduler = utils.instantiate(
            registry.scheduler, self.hparams.scheduler, optimizer
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": self.hparams.train.interval,  # 'epoch' or 'step'
            "monitor": self.hparams.train.monitor,
            "name": "trainer/lr",  # default is e.g. 'lr-AdamW'
        }
        # See documentation for how to configure the return
        # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_loader = self.dataset.train_dataloader(**self.hparams.loader)

        # Print stats in a try block since some dataloaders might not have a length?
        try:
            log.info(
                f"Loaded 'train' dataloader:".ljust(30) +
                f"{len(train_loader.dataset):7} examples | {len(train_loader):6} steps"
            )
        except:
            pass
        return train_loader

    def _eval_dataloaders_names(self, loaders, prefix):
        """Process loaders into a list of names and loaders"""
        if utils.is_dict(loaders):
            return [
                f"{prefix}/{k}" if k is not None else prefix for k in loaders.keys()
            ], list(loaders.values())
        elif utils.is_list(loaders):
            return [f"{prefix}/{i}" for i in range(len(loaders))], loaders
        else:
            return [prefix], [loaders]

    def _eval_dataloaders(self):
        # Return all val + test loaders
        val_loaders = self.dataset.val_dataloader(**self.hparams.loader)
        test_loaders = self.dataset.test_dataloader(**self.hparams.loader)
        val_loader_names, val_loaders = self._eval_dataloaders_names(val_loaders, "val")
        test_loader_names, test_loaders = self._eval_dataloaders_names(
            test_loaders, "test"
        )

        # Duplicate datasets for ema
        if self.hparams.train.ema > 0.0:
            val_loader_names += [name + "/ema" for name in val_loader_names]
            val_loaders = val_loaders + val_loaders
            test_loader_names += [name + "/ema" for name in test_loader_names]
            test_loaders = test_loaders + test_loaders

        # adding option to only have val loader at eval (eg if test is duplicate)
        if self.hparams.train.get("remove_test_loader_in_eval", None) is not None:
            eval_loader_names = val_loader_names
            eval_loaders = val_loaders
        # default behavior is to add test loaders in eval
        else:
            eval_loader_names = val_loader_names + test_loader_names
            eval_loaders = val_loaders + test_loaders

        return eval_loader_names, eval_loaders

    def val_dataloader(self):
        val_loader_names, val_loaders = self._eval_dataloaders()
        self.val_loader_names = val_loader_names
        try:
            for name, loader in zip(val_loader_names, val_loaders):
                log.info(
                    f"Loaded '{name}' dataloader:".ljust(30) +
                    f"{len(loader.dataset):7} examples | {len(loader):6} steps"
                )
        except:
            pass

        return val_loaders

    def test_dataloader(self):
        test_loader_names, test_loaders = self._eval_dataloaders()
        self.test_loader_names = ["final/" + name for name in test_loader_names]
        return test_loaders


### pytorch-lightning utils and entrypoint ###

def create_trainer(config):
    callbacks: List[pl.Callback] = []
    logger = None

    # WandB Logging
    if config.get("wandb") is not None:
        # Pass in wandb.init(config=) argument to get the nice 'x.y.0.z' hparams logged
        # Can pass in config_exclude_keys='wandb' to remove certain groups
        import wandb

        logger = CustomWandbLogger(
            config=utils.to_dict(config, recursive=True),
            settings=wandb.Settings(start_method="fork"),
            **config.wandb,
        )

    # Lightning callbacks
    if "callbacks" in config:
        for _name_, callback in config.callbacks.items():
            if callback is None: continue
            if config.get("wandb") is None and _name_ in ["learning_rate_monitor"]:
                continue
            log.info(f"Instantiating callback <{registry.callbacks[_name_]}>")
            callback._name_ = _name_
            callbacks.append(utils.instantiate(registry.callbacks, callback))

    # Profiler
    profiler = None
    if config.trainer.get("profiler", None) is not None:
        profiler = hydra.utils.instantiate(config.trainer.profiler)
        config.trainer.pop("profiler")


    # Configure ddp automatically
    if config.trainer.accelerator == 'gpu' and config.trainer.devices > 1:
        print("ddp automatically configured, more than 1 gpu used!")
        config.trainer.strategy = "ddp"
        
        # Enable find_unused_parameters for DML models to handle potential unused parameters
        if hasattr(config.model, 'output_head') and config.model.output_head == 'dml':
            print("DML model detected, enabling find_unused_parameters=True for DDP")
            config.trainer.strategy = "ddp_find_unused_parameters_true"

    # Add ProgressiveResizing callback
    if config.callbacks.get("progressive_resizing", None) is not None:
        num_stages = len(config.callbacks.progressive_resizing.stage_params)
        print(f"Progressive Resizing: {num_stages} stages")
        for i, e in enumerate(config.callbacks.progressive_resizing.stage_params):
            # Stage params are resolution and epochs, pretty print
            print(f"\tStage {i}: {e['resolution']} @ {e['epochs']} epochs")

    # Additional ModelCheckpoint callback for preemption
    if config.tolerance.id is not None:
        pass
        # if 'model_checkpoint' in config.callbacks.keys():
        #     callback_args = config.callbacks['model_checkpoint']
        #     callback_args._name_ = 'model_checkpoint'  # For the registry
        #     # Save last two checkpoints to be extra fault tolerant
        #     callback_args.save_top_k = 2
        #     callback_args.monitor = 'trainer/epoch'
        #     callback_args.mode = 'max'
        #     callback_args.save_last = False
        #     callback_args.filename = 'last'
        #     # callback_args.save_on_train_epoch_end = True # this is False for the other checkpoint callback
        #     ckpt_callback = utils.instantiate(registry.callbacks, callback_args)
        #     # ckpt_callback.CHECKPOINT_NAME_LAST = 'last_' # now we have two last checkpoints, last.ckpt and last_.ckpt
        #     callbacks.append(ckpt_callback)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        profiler=profiler,
        **config.trainer,
    )
    return trainer


def train(config):
    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)
    trainer = create_trainer(config)
    model = SequenceLightningModule(config)

    # Load pretrained_model if specified
    if config.train.get("pretrained_model_path", None) is not None:
        # PTL style.  Note, method returns a new model object, and need to pass config.
        model = SequenceLightningModule.load_from_checkpoint(
            config.train.pretrained_model_path,
            config=config,
            strict=config.train.pretrained_model_strict_load,
        )
        print("Loaded pretrained model from", config.train.pretrained_model_path)

        # Added by KS for pre-training
        # [22-07-21 AG] refactored, untested
        if config.train.get("ignore_pretrained_layers", False):
            pretrained_dict = model.state_dict()  # Use the loaded model
            model_dict = model.state_dict()
            for k, v in model_dict.items():
                for ignore_layer in config.train.ignore_pretrained_layers:
                    if ignore_layer in k:
                        pretrained_dict[k] = v
            model.load_state_dict(pretrained_dict)
        if config.train.get("pretrained_freeze_encoder", False):
            for name, param in model.named_parameters():
                if not("decoder" in name): param.requires_grad = False


    # Run initial validation epoch (useful for debugging, finetuning)
    if config.train.validate_at_start:
        print("Running validation before training")
        trainer.validate(model)

    if config.train.ckpt is not None:
        trainer.fit(model, ckpt_path=config.train.ckpt)
    else:
        trainer.fit(model)
    if config.train.test:
        trainer.test(model)



def preemption_setup(config):
    if config.tolerance.id is None:
        return config

    # Create path ./logdir/id/ to store information for resumption
    resume_dir = os.path.join(get_original_cwd(), config.tolerance.logdir, str(config.tolerance.id))

    if os.path.exists(resume_dir):
        print(f"Resuming from {resume_dir}")

        # Load path to the last checkpoint
        with open(os.path.join(resume_dir, "hydra.txt"), "r") as f:
            hydra_paths = list(f.readlines())

        # Look at the previous runs in reverse order
        checkpoint_path = None
        for hydra_path in reversed(hydra_paths):
            hydra_path = hydra_path.rstrip('\n')

            # Get the paths to the last.ckpt and last_.ckpt files
            last_path = os.path.join(hydra_path, "checkpoints", "last.ckpt")
            # last__path = os.path.join(hydra_path, "checkpoints", "last_.ckpt")
            # last_exists, last__exists = os.path.exists(last_path), os.path.exists(last__path)

            # if not last_exists or not last__exists:
            #     # This run doesn't have both checkpoints, so skip it
            #     print(f"\tSkipping {hydra_path}, not suitable for resuming (last_exists = {last_exists}, last__exists = {last__exists})")
            #     continue

            # # Read timestamp when checkpoints were modified
            # # We want to load the _earlier_ checkpoint, since that is guaranteed to be uncorrupted
            # last_timestamp = os.path.getmtime(last_path)
            # last__timestamp = os.path.getmtime(last__path)
            # print("\t\tlast_timestamp =", last_timestamp)
            # print("\t\tlast__timestamp =", last__timestamp)

            # if last_timestamp < last__timestamp:
            #     checkpoint_path = last_path
            # else:
            #     checkpoint_path = last__path
            # checkpoint_path = last_path
            # config.train.ckpt = checkpoint_path

            if os.path.exists(last_path):
                print("\tFound checkpoint at", last_path)
                config.train.ckpt = last_path
                # HACK TODO
                config.train.pretrained_model_path = None
                config.train.pretrained_model_state_hook._name_ = None
                # config.train.pretrained_model_reinit_hook._name_ = None
                break

        # If we didn't find a checkpoint
        if checkpoint_path is None:
            print("\tNo suitable checkpoint found, starting from scratch")

        # Set wandb run id to resume
        if os.path.exists(os.path.join(hydra_path, 'wandb')):
            run_info = [e for e in os.listdir(os.path.join(hydra_path, 'wandb')) if e.startswith('run-')][0]
            run_id = run_info.split('-')[-1]
            try:
                config.wandb.id = run_id
            except AttributeError:
                pass

    os.makedirs(resume_dir, exist_ok=True)

    # Store path to Hydra output folder
    with open(os.path.join(resume_dir, 'hydra.txt'), 'a') as f:
        f.write(os.getcwd() + '\n')

    return config


@hydra.main(config_path="configs", config_name="config.yaml")
def main(config: OmegaConf):

    # Process config:
    # - register evaluation resolver
    # - filter out keys used only for interpolation
    # - optional hooks, including disabling python warnings or debug friendly configuration
    config = utils.train.process_config(config)

    # Pretty print config using Rich library
    utils.train.print_config(config, resolve=True)

    config = preemption_setup(config)

    train(config)


if __name__ == "__main__":
    main()
