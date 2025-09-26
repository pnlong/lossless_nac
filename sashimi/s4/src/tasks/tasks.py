"""Implements Task interface, which consists of encoder + decoder + loss/metrics."""

from typing import Optional, List, Tuple
import math
import functools
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import ListConfig
from src.models.nn.normalization import (
    ReversibleInstanceNorm1dInput,
    ReversibleInstanceNorm1dOutput,
    TSNormalization,
    TSInverseNormalization,
)

from src.models.nn.adaptive_softmax import AdaptiveEmbedding, ProjectedAdaptiveLogSoftmax
import src.tasks.metrics as M
import src.models.nn.utils as U
import torchmetrics as tm
from src.utils.config import to_list, instantiate


class DummyMetric:
    """Dummy metric that returns 0.0 for DML models since torchmetrics don't work with dictionary outputs."""

    def __init__(self):
        pass

    def update(self, *args, **kwargs):
        """Do nothing - DML doesn't use torchmetrics."""
        pass

    def compute(self):
        """Return 0.0 as a placeholder."""
        return 0.0

    def reset(self):
        """Do nothing."""
        pass


class BaseTask:
    """Abstract class for all tasks.

    This class takes care of:
    - loss function
    - arbitrary metrics
    - (optional) encoder module that interfaces with dataset (inputs) and model
    - (optional) decoder module that interfaces with dataset (targets) and model
    """
    encoder = None
    decoder = None

    def __init__(self, dataset=None, model=None, loss=None, loss_val=None, metrics=None, torchmetrics=None):
        # print(f"🔍 DEBUG: BaseTask.__init__ called")
        # print(f"🔍 DEBUG: BaseTask parameters:")
        # print(f"🔍 DEBUG:   - dataset: {type(dataset).__name__ if dataset else None}")
        # print(f"🔍 DEBUG:   - model: {type(model).__name__ if model else None}")
        # print(f"🔍 DEBUG:   - loss: {loss}")
        # print(f"🔍 DEBUG:   - loss_val: {loss_val}")
        # print(f"🔍 DEBUG:   - metrics: {metrics}")
        # print(f"🔍 DEBUG:   - torchmetrics: {torchmetrics}")
        
        """ This class is allowed to grab attributes directly off a constructed dataset and model object """
        self.dataset = dataset
        self.model = model
        if metrics is None: metrics = []
        self.metric_names = to_list(metrics)

        if torchmetrics is None: torchmetrics = []
        
        # print(f"🔍 DEBUG: BaseTask setup_stereo_embedding about to be called")
        # Add stereo embedding support for audio datasets
        # Removed _setup_stereo_embedding() - let LMTask handle embedding for both mono and stereo
        # print(f"🔍 DEBUG: BaseTask setup_stereo_embedding completed")
        
        self.torchmetric_names = to_list(torchmetrics)
        self._tracked_torchmetrics = {}

        # The decoder might pass through arguments that the loss needs (e.g. sequence lengths)
        # but might also pass through extraneous arguments (e.g. sampling rate)
        # Wrap loss and metrics so that they accept kwargs and

        # Create loss function
        # Check for DML loss BEFORE instantiation (loss is still the config dict/string)
        is_dml_loss = False
        if isinstance(loss, dict) and loss.get('_name_') == 'dml':
            is_dml_loss = True
        elif isinstance(loss, str) and loss == 'dml':
            is_dml_loss = True

        self.loss = instantiate(M.output_metric_fns, loss, partial=True)

        # Special handling for DML loss which needs dataset parameter
        if is_dml_loss:
            # Create closure that captures dataset for DML loss
            dml_loss_fn = self.loss
            self.loss = lambda preds, targets, **kwargs: dml_loss_fn(preds, targets, dataset=self.dataset)
        else:
            self.loss = U.discard_kwargs(self.loss)

        if loss_val is not None:
            # Check for DML loss_val BEFORE instantiation
            is_dml_loss_val = False
            if isinstance(loss_val, dict) and loss_val.get('_name_') == 'dml':
                is_dml_loss_val = True
            elif isinstance(loss_val, str) and loss_val == 'dml':
                is_dml_loss_val = True

            self.loss_val = instantiate(M.output_metric_fns, loss_val, partial=True)

            # Special handling for DML loss_val
            if is_dml_loss_val:
                # Create closure that captures dataset for DML loss_val
                dml_loss_val_fn = self.loss_val
                self.loss_val = lambda preds, targets, **kwargs: dml_loss_val_fn(preds, targets, dataset=self.dataset)
            else:
                self.loss_val = U.discard_kwargs(self.loss_val)


    def _init_torchmetrics(self, prefix):
        """Instantiate torchmetrics."""
        # TODO torchmetrics is better renamed to "epoch_metrics" or something

        self._tracked_torchmetrics[prefix] = {}
        for name in self.torchmetric_names:
            # Skip torchmetrics for DML since it outputs dictionaries, not classification logits
            if hasattr(self, 'model') and hasattr(self.model, 'output_head_type') and self.model.output_head_type == 'dml':
                # For DML, create dummy metrics that return 0
                self._tracked_torchmetrics[prefix][name] = DummyMetric()
            else:
                # Normal torchmetrics initialization for categorical
                if name in ['AUROC', 'StatScores', 'Precision', 'Recall', 'F1', 'F1Score']:
                    self._tracked_torchmetrics[prefix][name] = getattr(tm, name)(average='macro', num_classes=self.dataset.d_output, compute_on_step=False).to('cuda')
                elif '@' in name:
                    k = int(name.split('@')[1])
                    mname = name.split('@')[0]
                    self._tracked_torchmetrics[prefix][name] = getattr(tm, mname)(average='macro', num_classes=self.dataset.d_output, compute_on_step=False, top_k=k).to('cuda')
                else:
                    self._tracked_torchmetrics[prefix][name] = getattr(tm, name)(compute_on_step=False).to('cuda')

    def _reset_torchmetrics(self, prefix=None):
        """Reset torchmetrics for a prefix associated with a particular dataloader (e.g. train, val, test).

        Generally do this at the start of an epoch.
        """
        all_prefixes = [prefix] if prefix is not None else self._tracked_torchmetrics
        for prefix in all_prefixes:
            for name in self.torchmetric_names:
                try:
                    self._tracked_torchmetrics[prefix][name].reset()
                except KeyError:  # metrics don't exist yet
                    pass

    def get_torchmetrics(self, prefix):
        """Compute torchmetrics for a prefix associated with a particular dataloader (e.g. train, val, test).

        Generally do this at the end of an epoch.
        """
        return {name: self._tracked_torchmetrics[prefix][name].compute() for name in self.torchmetric_names}

    def torchmetrics(self, x, y, prefix):
        """Update torchmetrics with new data.

        Prefix corresponds to a particular dataloader (e.g. train, val, test).

        Generally call this every batch.
        """
        if prefix not in self._tracked_torchmetrics:
            self._init_torchmetrics(prefix)

        for name in self.torchmetric_names:
            if name.startswith('Accuracy'):
                if len(x.shape) > 2:
                    # Multi-dimensional, multi-class
                    self._tracked_torchmetrics[prefix][name].update(x.transpose(1, 2), y.squeeze())
                    continue
            self._tracked_torchmetrics[prefix][name].update(x, y)

    def metrics(self, x, y, **kwargs):
        """Add metrics to the task.

        Metrics are just functions:
        - output metrics are a function of output and target
        - loss metrics are a function of loss (e.g. perplexity)
        """
        output_metrics = {
            name: U.discard_kwargs(M.output_metric_fns[name])(x, y, **kwargs)
            for name in self.metric_names if name in M.output_metric_fns
        }
        loss_metrics = {
            name: U.discard_kwargs(M.loss_metric_fns[name])(x, y, self.loss, **kwargs)
            for name in self.metric_names if name in M.loss_metric_fns
        }
        return {**output_metrics, **loss_metrics}


class Scalar(nn.Module):
    def __init__(self, c=1):
        super().__init__()
        self.c = c
    def forward(self, x):
        return x * self.c

class LMTask(BaseTask):
    def __init__(self, tied=False, rescale=True, **kwargs):
        # print(f"🔍 DEBUG: LMTask.__init__ called with kwargs: {kwargs}")
        super().__init__(loss='cross_entropy', **kwargs)
        
        # print(f"🔍 DEBUG: LMTask initialization parameters:")
        n_tokens = self.dataset.n_tokens
        d_model = self.model.d_model
        d_output = self.model.d_output
        d_input = self.dataset.d_input
        
        # print(f"🔍 DEBUG:   - n_tokens: {n_tokens}")
        # print(f"🔍 DEBUG:   - d_model: {d_model}")
        # print(f"🔍 DEBUG:   - d_output: {d_output}")
        # print(f"🔍 DEBUG:   - d_input: {d_input}")
        # print(f"🔍 DEBUG:   - tied: {tied}")
        # print(f"🔍 DEBUG:   - rescale: {rescale}")

        if rescale:
            scale = Scalar(math.sqrt(d_model))
            # print(f"🔍 DEBUG:   - scale factor: {math.sqrt(d_model)}")
        else:
            scale = None
            # print(f"🔍 DEBUG:   - scale factor: None")

        # Pass stereo information to the model so it knows how to handle output reshaping
        if hasattr(self.dataset, 'is_stereo') and self.dataset.is_stereo:
            # Set stereo flag on the model for output handling
            if hasattr(self.model, 'is_stereo'):
                self.model.is_stereo = True
            if hasattr(self.model, 'interleaving_strategy'):
                self.model.interleaving_strategy = getattr(self.dataset, 'interleaving_strategy', 'temporal')
            print(f"🔍 DEBUG: Model configured for stereo processing with strategy: {getattr(self.dataset, 'interleaving_strategy', 'temporal')}")

        # Check if stereo encoder was already set up by BaseTask
        print(f"🔍 DEBUG: LMTask - hasattr(self, 'encoder'): {hasattr(self, 'encoder')}")
        if hasattr(self, 'encoder'):
            print(f"🔍 DEBUG: LMTask - self.encoder is not None: {self.encoder is not None}")
            if self.encoder is not None:
                print(f"🔍 DEBUG: LMTask - existing encoder type: {type(self.encoder).__name__}")
        
        if hasattr(self, 'encoder') and self.encoder is not None:
            print(f"🔍 DEBUG: ✅ Encoder already set up by BaseTask (stereo case) - using existing encoder")
            print(f"🔍 DEBUG: Existing encoder type: {type(self.encoder).__name__}")
            # Don't overwrite the existing encoder - just add scaling if needed
            if rescale:
                print(f"🔍 DEBUG: Adding scaling to existing encoder")
                # Wrap existing encoder with scaling
                self.encoder = U.PassthroughSequential(
                    self.encoder,
                    scale,
                )
            print(f"🔍 DEBUG: Final encoder components: {[type(comp).__name__ for comp in self.encoder]}")
        else:
            # print(f"🔍 DEBUG: ❌ No existing encoder - creating new encoder (mono case)")
            # Handle multi-channel inputs (stereo) - this should not happen if stereo is enabled
            # print(f"🔍 DEBUG: Checking if d_input > 1: {d_input > 1}")
            if d_input > 1:
                # print(f"🔍 DEBUG: ⚠️ WARNING: Multi-channel case but no stereo encoder set up!")
                # print(f"🔍 DEBUG: ✅ Multi-channel case (stereo) - creating MultiChannelEmbedding")
                # For stereo: create separate embeddings for each channel
                embeddings = []
                for i in range(d_input):
                    # print(f"🔍 DEBUG: Creating embedding {i+1}/{d_input} with shape ({n_tokens}, {d_model})")
                    emb = nn.Embedding(n_tokens, d_model)
                    nn.init.normal_(emb.weight, mean=0, std=d_model**-.5)
                    embeddings.append(emb)

                # Combine embeddings with channel-wise processing
                class MultiChannelEmbedding(nn.Module):
                    def __init__(self, embeddings):
                        super().__init__()
                        # print(f"🔍 DEBUG: MultiChannelEmbedding.__init__ called with {len(embeddings)} embeddings")
                        self.embeddings = nn.ModuleList(embeddings)

                    def forward(self, x):
                        # print(f"🔍 DEBUG: MultiChannelEmbedding.forward called with input shape: {x.shape}")
                        # x shape: (batch, length, channels)
                        batch_size, seq_len, channels = x.shape
                        # print(f"🔍 DEBUG: MultiChannelEmbedding input breakdown: batch={batch_size}, seq_len={seq_len}, channels={channels}")
                        
                        embedded_channels = []
                        for c in range(channels):
                            # print(f"🔍 DEBUG: Processing channel {c+1}/{channels}")
                            channel_input = x[:, :, c]  # (batch, length)
                            # print(f"🔍 DEBUG: Channel {c+1} input shape: {channel_input.shape}")
                            channel_emb = self.embeddings[c](channel_input)  # (batch, length, d_model)
                            # print(f"🔍 DEBUG: Channel {c+1} embedded shape: {channel_emb.shape}")
                            embedded_channels.append(channel_emb)

                        # Concatenate along feature dimension: (batch, length, channels * d_model)
                        result = torch.cat(embedded_channels, dim=-1)
                        # print(f"🔍 DEBUG: MultiChannelEmbedding final output shape: {result.shape}")
                        # print(f"🔍 DEBUG: Expected output shape: (batch={batch_size}, length={seq_len}, features={channels * d_model})")
                        return result

                embedding = MultiChannelEmbedding(embeddings)
                # print(f"🔍 DEBUG: MultiChannelEmbedding created successfully")
                # Note: d_model adjustment is handled in train.py via embedded_d_input calculation
            else:
                # Mono case: use registry-based embedding with squeeze
                from src.tasks.encoders import registry
                embedding = registry["embedding"](n_tokens, d_model)

            encoder = U.PassthroughSequential(
                embedding,
                scale,
            )
            self.encoder = encoder

        # For stereo, d_output might be per-channel, so adjust decoder accordingly
        decoder_input_dim = d_output
        # print(f"🔍 DEBUG: Decoder input dimension calculation:")
        # print(f"🔍 DEBUG:   - Initial decoder_input_dim: {decoder_input_dim}")
        # print(f"🔍 DEBUG:   - d_input: {d_input}")
        # print(f"🔍 DEBUG:   - d_output: {d_output}")
        
        if d_input > 1:
            # For stereo, model outputs predictions for each channel
            # Decoder should output per-channel predictions
            decoder_input_dim = d_output // d_input if d_output > d_input else d_output
            # print(f"🔍 DEBUG:   - Stereo case: decoder_input_dim = {decoder_input_dim}")

        # Always create linear decoder - backbone never applies output head
        decoder = nn.Linear(decoder_input_dim, n_tokens)

        if tied and d_input == 1:
            # Apply weight tying between encoder and decoder
            assert d_model == d_output
            decoder.weight = self.encoder[0].weight
        # else:
        #     print(f"🔍 DEBUG: Tied weights disabled (tied={tied}, d_input={d_input})")
            
        self.decoder = decoder
        # print(f"🔍 DEBUG: LMTask initialization completed successfully")

class ForecastingTask(BaseTask):

    class DummyModule(nn.Module):

        def forward(self, *args):
            return args

    def __init__(self, norm='mean', **kwargs):
        # print(f"🔍 DEBUG: ForecastingTask.__init__ called with norm='{norm}', kwargs: {kwargs}")
        super().__init__(**kwargs)

        # print(f"🔍 DEBUG: ForecastingTask normalization setup:")
        # print(f"🔍 DEBUG:   - norm parameter: {norm}")
        
        if norm == 'revnorm':
            # print(f"🔍 DEBUG: Using ReversibleInstanceNorm1dInput/Output")
            # For normalization, use the feature dimension after embedding
            # For stereo: d_model has already been scaled by d_input in the embedding section
            d_model = self.model.d_model
            # print(f"🔍 DEBUG:   - d_model for normalization: {d_model}")
            self.encoder = ReversibleInstanceNorm1dInput(d_model, transposed=False)
            self.decoder = ReversibleInstanceNorm1dOutput(self.encoder)
            # print(f"🔍 DEBUG:   - ReversibleInstanceNorm1dInput/Output created")
        elif norm == 'mean':
            self.encoder = TSNormalization(method='mean', horizon=self.dataset.dataset_train.forecast_horizon)
            self.decoder = TSInverseNormalization(method='mean', normalizer=self.encoder)
        elif norm == 'last':
            self.encoder = TSNormalization(method='last', horizon=self.dataset.dataset_train.forecast_horizon)
            self.decoder = TSInverseNormalization(method='last', normalizer=self.encoder)
        else:
            self.encoder = None
            self.decoder = None

        try:
            if hasattr(self.dataset.dataset_train, 'mean'):
                self.mean = torch.tensor(self.dataset.dataset_train.mean)
                self.std = torch.tensor(self.dataset.dataset_train.std)
            elif hasattr(self.dataset.dataset_train, 'standardization'):
                self.mean = torch.tensor(self.dataset.dataset_train.standardization['means'])
                self.std = torch.tensor(self.dataset.dataset_train.standardization['stds'])
            else:
                self.mean = None
                self.std = None
        except AttributeError:
            raise AttributeError('Dataset does not have mean/std attributes')
            self.mean = torch.tensor(self.dataset.dataset_train.standardization['means'])
            self.std = torch.tensor(self.dataset.dataset_train.standardization['stds'])

        if hasattr(self.dataset.dataset_train, 'log_transform'):
            self.log_transform = self.dataset.dataset_train.log_transform
        else:
            self.log_transform = False
        print("Log Transform", self.log_transform)

    def metrics(self, x, y, state=None, timestamps=None, ids=None): # Explicit about which arguments the decoder might pass through, but can future-proof with **kwargs
        if self.mean is not None:
            means = self.mean[ids].to(x.device)
            stds = self.std[ids].to(x.device)
            x_ = x * stds[:, None, None] + means[:, None, None]
            y_ = y * stds[:, None, None] + means[:, None, None]
        else:
            x_ = x
            y_ = y

        if self.log_transform:
            x_ = torch.exp(x_)
            y_ = torch.exp(y_)

        return super().metrics(x_, y_)

class VideoTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self._y_to_logits = {}
        self._vid_to_logits = {}
        self._vid_to_label = {}

        # TODO needed to extract the first element of y, which includes the video idea; there should be a cleaner pattern to this
        import copy
        loss_fn = copy.deepcopy(self.loss)
        self.loss = lambda x, y: loss_fn(x, y[0])
        self.loss = U.discard_kwargs(self.loss)  # remove extra kwargs
        if hasattr(self, 'loss_val'):
            loss_val_fn = copy.deepcopy(self.loss_val)
            self.loss_val = lambda x, y: loss_val_fn(x, y[0])
            self.loss_val = U.discard_kwargs(self.loss_val)  # remove extra kwargs

    def metrics(self, logits, y, **kwargs):
        labels, vids = y
        return super().metrics(logits, labels, **kwargs)

    def torchmetrics(self, logits, y, prefix):
        """
        logits: (batch, n_classes)
        y = tuple of labels and video ids
        labels: (batch)
        vids: (batch)
        """
        for _logits, _label, _vid in zip(logits, y[0], y[1]):
            _vid = _vid.item()
            # Check that labels are consistent per video id
            assert self._vid_to_label[prefix].get(_vid, _label) == _label
            self._vid_to_label[prefix][_vid] = _label

            self._vid_to_logits[prefix][_vid].append(_logits)

    def _reset_torchmetrics(self, prefix):
        self._vid_to_logits[prefix] = collections.defaultdict(list)
        self._vid_to_label[prefix] = {}

    def get_torchmetrics(self, prefix):
        vid_to_average_logits = {vid: torch.mean(torch.stack(logits, dim=0), dim=0) for vid, logits in self._vid_to_logits[prefix].items()}
        # y is (label, vid) pair
        all_labels = torch.stack(list(self._vid_to_label[prefix].values()), dim=0) # (n_videos)
        all_logits = torch.stack(list(vid_to_average_logits.values()), dim=0) # (n_videos, n_classes)
        m = M.accuracy(all_logits, all_labels)
        return {'aggregate_accuracy': m}


class AdaptiveLMTask(BaseTask):
    def __init__(
        self,
        div_val,
        cutoffs : List[int],
        tie_weights : bool,
        tie_projs : List[bool],
        init_scale=1.0,
        bias_scale=0.0,
        dropemb=0.0,
        dropsoft=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        n_tokens = self.dataset.n_tokens
        d_model = self.model.d_model
        d_output = self.model.d_output

        encoder = AdaptiveEmbedding(
            n_tokens,
            d_model,
            d_model,
            cutoffs=cutoffs,
            div_val=div_val,
            init_scale=init_scale,
            dropout=dropemb,
        )

        if tie_weights:
            assert d_model == d_output
            emb_layers = [i.weight for i in encoder.emb_layers]
        else:
            emb_layers = None

        # Construct decoder/loss
        emb_projs = encoder.emb_projs
        loss = ProjectedAdaptiveLogSoftmax(
            n_tokens, d_output, d_output,
            cutoffs, div_val=div_val,
            tie_projs=tie_projs,
            out_projs=emb_projs,
            out_layers_weights=emb_layers,
            bias_scale=bias_scale,
            dropout=dropsoft,
        )

        self.encoder = encoder
        self.loss = loss


class ImageNetTask(BaseTask):
    """
    Imagenet training uses mixup augmentations, which require a separate loss for train and val,
    which we overide the base task here.

    Not really used anymore.
    """

    def __init__(self, **kwargs):
        import hydra

        super().__init__(
            dataset=kwargs.get("dataset", None),
            model=kwargs.get("model", None),
            loss=kwargs.get("loss", None),  # we still create the base loss here, but will overide below
            metrics=kwargs.get("metrics", None),
            torchmetrics=kwargs.get("torchmetrics", None)
        )

        # if using mixup, overide loss (train) and loss_val, otherwise
        # we have just one loss from the base task above
        if "loss_val" in kwargs and "loss_train" in kwargs:
            self.loss = hydra.utils.instantiate(kwargs.get("loss_train"))
            self.loss_val = hydra.utils.instantiate(kwargs.get('loss_val'))


registry = {
    'base': BaseTask,
    'lm': LMTask,
    'adaptivelm': AdaptiveLMTask,
    'imagenet': ImageNetTask,
    'forecasting': ForecastingTask,
    'video': VideoTask,
}
