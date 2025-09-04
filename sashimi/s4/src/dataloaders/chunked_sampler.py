"""ChunkedSampler for cycling through dataset in configurable chunks."""

import os
import numpy as np
import torch
from torch.utils.data import Sampler


class ChunkedSampler(Sampler):
    """
    A PyTorch Sampler that yields dataset indices in configurable chunks,
    with automatic cycling through the entire dataset.

    This sampler allows training on chunks of the dataset while ensuring
    the entire dataset is covered over multiple epochs.
    """

    def __init__(self, dataset_size: int, chunk_size: int, seed: int = 42, debug: bool = False):
        """
        Initialize the ChunkedSampler.

        Args:
            dataset_size: Total size of the dataset
            chunk_size: Number of samples per chunk
            seed: Random seed for reproducible shuffling
            debug: Enable debug output
        """
        self.dataset_size = dataset_size
        self.chunk_size = chunk_size
        self.current_chunk = 0
        self.rng = np.random.RandomState(seed)
        self.debug = debug or os.getenv('CHUNKED_SAMPLER_DEBUG', 'false').lower() == 'true'

        # Calculate total number of chunks needed to cover the dataset
        self.total_chunks = (dataset_size + chunk_size - 1) // chunk_size

        # if self.debug:
        #     print(f"🔧 ChunkedSampler: {self.total_chunks} chunks total")

    def get_chunk_indices(self, chunk_idx: int) -> list:
        """
        Get indices for a specific chunk, cycling through the dataset.

        Args:
            chunk_idx: Index of the chunk to generate

        Returns:
            List of dataset indices for this chunk
        """
        # Calculate start position for this chunk (with wrapping)
        start_idx = (chunk_idx * self.chunk_size) % self.dataset_size
        end_idx = min(start_idx + self.chunk_size, self.dataset_size)

        if end_idx < start_idx + self.chunk_size:
            # Wrap around to beginning of dataset
            remaining = (start_idx + self.chunk_size) - self.dataset_size
            indices = list(range(start_idx, self.dataset_size)) + list(range(remaining))
        else:
            # No wrapping needed
            indices = list(range(start_idx, end_idx))

        # Shuffle the indices within this chunk for randomness
        indices = self.rng.permutation(indices).tolist()

        return indices

    def set_chunk(self, chunk_idx: int):
        """
        Set the current chunk index.

        Args:
            chunk_idx: Chunk index to set
        """
        self.current_chunk = chunk_idx % self.total_chunks

    def advance_chunk(self):
        """Advance to the next chunk (with wrapping)."""
        self.current_chunk = (self.current_chunk + 1) % self.total_chunks

    def __iter__(self):
        """Return iterator over indices for the current chunk."""
        return iter(self.get_chunk_indices(self.current_chunk))

    def __len__(self):
        """Return the size of the current chunk."""
        return min(self.chunk_size, self.dataset_size)

    def get_chunk_info(self) -> dict:
        """
        Get information about the current chunk.

        Returns:
            Dictionary with chunk information
        """
        return {
            'current_chunk': self.current_chunk,
            'total_chunks': self.total_chunks,
            'chunk_size': self.chunk_size,
            'dataset_size': self.dataset_size,
            'samples_in_current_chunk': len(self)
        }

    def reset(self):
        """Reset to the first chunk."""
        self.current_chunk = 0
