# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains a language model on the Enwik8 dataset."""

import functools
import random
from typing import Any, List, Optional

from absl import app
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import tree

from language_modeling_is_compression import constants
from language_modeling_is_compression import data_loaders
from language_modeling_is_compression import transformer


def _to_marginals(
    predictions: jax.Array,
    sequences: jax.Array,
) -> jax.Array:
  """Converts a conditional array to a marginals array."""
  true_predictions = jnp.take_along_axis(
      predictions, sequences[..., None], axis=-1
  )
  true_predictions = true_predictions[..., 0]  # Shape (B, T).
  return jnp.sum(true_predictions, axis=1)  # Shape (B,).


def _make_loss_fn(model: hk.Transformed) -> Any:
  """Returns the loss function for update_parameters."""

  def loss_fn(
      params: hk.Params,
      sequences: jax.Array,
  ) -> jnp.float32:
    """Returns the loss for the model and the last state.

    Args:
      params: The parameters of the model, usually a neural network.
      sequences: The input of sequences to evaluate. See neural_predictors.py.
    """
    conditionals = model.apply(
        params=params,
        targets=sequences,
        rng=None,
    )
    marginals = _to_marginals(conditionals, sequences)
    return -jnp.mean(marginals)

  return loss_fn


@functools.partial(
    jax.jit, static_argnames=('optimizer', 'grad_fn', 'normalize_gradients')
)
def _update_parameters(
    params: hk.Params,
    opt_state: optax.OptState,
    sequences: jax.Array,
    grad_fn: Any,
    optimizer: optax.GradientTransformation,
    normalize_gradients: bool = True,
) -> tuple[hk.Params, optax.OptState, dict[str, Any]]:
  """Returns updated params and extra logs (like loss, last state etc).

  Backpropagation is done on the whole sequence. The whole function is jitted.

  Args:
    params: The current parameters of the network.
    opt_state: The optimizer state.
    sequences: The input of sequences to evaluate. See base_predictor.py.
    grad_fn: A gradient function, which takes some parameters, a random seed,
      the data to compute the gradient on, and an initial state for the
      predictor. It returns the gradient of the parameters for this batch of
      data, and extra values.
    optimizer: An optax optimizer.
    normalize_gradients: Whether to divide the gradients by the length of the
      sequences, or keep them as is. Using this option guarantees to have the
      same scale across various sequence lengths, and therefore tasks.
  """
  loss, grad = grad_fn(params, sequences)
  if normalize_gradients:
    length_sequence = float(sequences.shape[1])
    grad = tree.map_structure(lambda x: x / length_sequence, grad)
  updates, new_opt_state = optimizer.update(grad, opt_state)
  new_params = optax.apply_updates(params, updates)

  log_dict = {
      'loss': loss,
      'grad_norm_unclipped': optax.global_norm(grad),
  }

  return new_params, new_opt_state, log_dict


def train_transformer_decoder(
    training_steps: int,
    log_every: int,
    batch_size: int = 128,
    sequence_length: int = constants.CHUNK_SIZE_BYTES,
    use_tqdm: bool = True,
) -> tuple[hk.Params, float]:
  """Trains a language model on Enwik8 data.

  Sequences of 2048 characters are extracted from Enwik8, and then randomly
  sampled. We train a decoder-only transformer on batches, minimizing the
  log-loss objective. The exact architecture can be modified using the
  TransformerConfig object (defined in transformer.py)

  Args:
    training_steps: Number of batches to train on.
    log_every: How often to log the loss. If negative or 0, no log at all.
    batch_size: The number of sequences in a batch.
    sequence_length: The length of the sequences to train on, in number of ASCII
      characters.
    use_tqdm: Whether to use a progress bar or not.

  Returns:
    The final loss, and final parameters.
  """
  config = transformer.TransformerConfig(vocab_size=constants.ALPHABET_SIZE)
  model = hk.transform(
      functools.partial(transformer.transformer_decoder, config=config)
  )

  data_generator = data_loaders.get_enwik9_iterator(
      # Divide by 10 to only fetch Enwik8 data.
      num_chunks=constants.NUM_CHUNKS // 10,
      sequence_length=sequence_length,
  )
  dataset = list(data_generator)

  def fetch_random_batch() -> np.ndarray:
    batch_list = random.choices(dataset, k=batch_size)
    batch_list = [np.frombuffer(seq, dtype=np.uint8) for seq in batch_list]
    return np.array(batch_list, dtype=np.uint8)

  # Initialize parameters.
  dummy_batch = fetch_random_batch()
  rng = jax.random.PRNGKey(0)
  params = model.init(rng, dummy_batch)

  # Make gradient function.
  loss_fn = _make_loss_fn(model)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=False)

  # Make optimizer, to apply the gradients.
  optimizer = optax.adam(learning_rate=1e-4)
  opt_state = optimizer.init(params)

  logging.info('Initialization done, starting training...')
  last_loss = 0.0
  for step in tqdm.trange(training_steps, disable=not use_tqdm):
    batch = fetch_random_batch()
    logging.info('Batch fetched.')

    params, opt_state, logs = _update_parameters(
        params=params,
        opt_state=opt_state,
        sequences=batch,
        grad_fn=grad_fn,
        optimizer=optimizer,
    )
    if log_every > 0 and step % log_every == 0:
      logging.info(
          'Step %f, Loss %f, Grad norm %f',
          step,
          logs['loss'],
          logs['grad_norm_unclipped'],
      )
    last_loss = logs['loss']

  return params, last_loss


def train_audio_transformer(
    audio_files: List[str],
    use_16bit: bool = False,
    blocking_size: int = 1024,
    training_steps: int = 1000,
    log_every: int = 100,
    batch_size: int = 128,
    sequence_length: int = constants.CHUNK_SIZE_BYTES,
    use_tqdm: bool = True,
    wandb_project: Optional[str] = None,
    wandb_group_name: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_config: Optional[dict] = None,
) -> tuple[hk.Params, float]:
    """Train audio transformer with wandb logging support.
    
    Args:
        audio_files: List of paths to WAV files (provided by calling code)
        use_16bit: Whether to use 16-bit audio (affects vocabulary size)
        blocking_size: Size of blocks for stereo processing
        training_steps: Number of training steps
        log_every: How often to log metrics
        batch_size: Batch size for training
        sequence_length: Length of sequences in bytes
        use_tqdm: Whether to show progress bar
        wandb_project: Wandb project name (if None, no wandb logging)
        wandb_group_name: Wandb group name for organizing runs
        wandb_run_name: Wandb run name
        wandb_config: Additional config to log to wandb
        
    Returns:
        Tuple of (final_params, final_loss)
    """
    
    # Initialize wandb if project is specified
    if wandb_project is not None:
        try:
            import wandb
        except ImportError:
            logging.warning("wandb not installed, skipping wandb logging")
            wandb_project = None
    
    if wandb_project is not None:
        import wandb
        wandb_config = wandb_config or {}
        wandb_config.update({
            'use_16bit': use_16bit,
            'blocking_size': blocking_size,
            'training_steps': training_steps,
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'num_audio_files': len(audio_files),
        })
        
        wandb.init(
            project=wandb_project,
            group=wandb_group_name,
            name=wandb_run_name,
            config=wandb_config
        )
        
        # Log audio file info
        wandb.log({
            'dataset/num_files': len(audio_files),
            'dataset/use_16bit': use_16bit,
            'dataset/blocking_size': blocking_size,
        })
    
    # Validate configuration
    data_loaders.validate_audio_config(use_16bit, blocking_size, sequence_length)
    
    # Create model configuration
    config = transformer.create_audio_transformer_config(use_16bit=use_16bit)
    model = hk.transform(
        functools.partial(transformer.transformer_decoder, config=config)
    )
    
    # Create data generator
    data_generator = data_loaders.get_custom_audio_iterator(
        audio_files=audio_files,
        num_chunks=constants.NUM_CHUNKS,
        use_16bit=use_16bit,
        blocking_size=blocking_size,
        chunk_size_bytes=sequence_length,
    )
    dataset = list(data_generator)
    
    # Log dataset statistics
    if wandb_project is not None:
        wandb.log({
            'dataset/total_chunks': len(dataset),
            'dataset/chunk_size_bytes': sequence_length,
            'dataset/vocab_size': config.vocab_size,
        })
    
    def fetch_random_batch() -> np.ndarray:
        batch_list = random.choices(dataset, k=batch_size)
        if use_16bit:
            # For 16-bit, convert bytes to int16 arrays
            batch_list = [np.frombuffer(seq, dtype=np.int16) for seq in batch_list]
        else:
            # For 8-bit, convert bytes to uint8 arrays
            batch_list = [np.frombuffer(seq, dtype=np.uint8) for seq in batch_list]
        return np.array(batch_list, dtype=np.int16 if use_16bit else np.uint8)
    
    # Initialize parameters
    dummy_batch = fetch_random_batch()
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, dummy_batch)
    
    # Log model parameters
    if wandb_project is not None:
        total_params = sum(x.size for x in jax.tree_leaves(params))
        wandb.log({
            'model/total_parameters': total_params,
            'model/vocab_size': config.vocab_size,
            'model/embedding_dim': config.embedding_dim,
            'model/num_layers': config.num_layers,
            'model/num_heads': config.num_heads,
        })
    
    # Make gradient function
    loss_fn = _make_loss_fn(model)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    
    # Make optimizer
    optimizer = optax.adam(learning_rate=1e-4)
    opt_state = optimizer.init(params)
    
    # Log optimizer config
    if wandb_project is not None:
        wandb.log({
            'optimizer/learning_rate': 1e-4,
            'optimizer/optimizer': 'adam',
        })
    
    logging.info('Initialization done, starting training...')
    last_loss = 0.0
    
    for step in tqdm.trange(training_steps, disable=not use_tqdm):
        batch = fetch_random_batch()
        
        params, opt_state, logs = _update_parameters(
            params=params,
            opt_state=opt_state,
            sequences=batch,
            grad_fn=grad_fn,
            optimizer=optimizer,
        )
        
        # Log metrics
        if log_every > 0 and step % log_every == 0:
            logging.info(
                'Step %f, Loss %f, Grad norm %f',
                step,
                logs['loss'],
                logs['grad_norm_unclipped'],
            )
            
            # Log to wandb
            if wandb_project is not None:
                wandb.log({
                    'train/step': step,
                    'train/loss': float(logs['loss']),
                    'train/grad_norm': float(logs['grad_norm_unclipped']),
                    'train/learning_rate': 1e-4,  # Could be made configurable
                })
        
        last_loss = logs['loss']
    
    # Log final metrics
    if wandb_project is not None:
        wandb.log({
            'train/final_loss': float(last_loss),
            'train/total_steps': training_steps,
        })
        
        # Save model parameters as wandb artifact
        np.savez('params.npz', **params)
        artifact = wandb.Artifact('model_params', type='model')
        artifact.add_file('params.npz')
        wandb.log_artifact(artifact)
        
        wandb.finish()
    
    return params, last_loss


def main(_) -> None:
  """Trains a language model and saves the parameters to a JSON file."""
  params, loss = train_transformer_decoder(
      training_steps=100,
      log_every=10,
      sequence_length=constants.CHUNK_SIZE_BYTES,
  )
  logging.info('Final loss: %f', loss)

  np.savez('params.npz', **params)
  logging.info('Parameters saved in file params.npz')


if __name__ == '__main__':
  app.run(main)
