"""ChunkedDataLoader wrapper for automatic chunking detection."""

import os
import torch
from torch.utils.data import DataLoader
from .chunked_sampler import ChunkedSampler


class ChunkedDataLoader(torch.utils.data.DataLoader):
    """
    A PyTorch DataLoader subclass that provides automatic chunking functionality.

    This wrapper automatically detects when chunking should be enabled based on
    the chunk_size parameter, and falls back to standard DataLoader behavior
    when chunking is not needed.
    """

    def __init__(self, dataset, batch_size, chunk_size=None, current_chunk=None, dataloader_type=None, **dataloader_kwargs):
        """
        Initialize the ChunkedDataLoader.

        Args:
            dataset: The dataset to load from
            batch_size: Size of each batch
            chunk_size: Size of each chunk (if None, uses standard DataLoader)
            current_chunk: Starting chunk index (if None, starts from 0)
            dataloader_type: Type of dataloader ('train' or 'eval') for chunk syncing
            **dataloader_kwargs: Additional arguments for DataLoader
        """
        # Store dataset reference first
        self._original_dataset = dataset
        self._dataloader_type = dataloader_type  # Store dataloader type for syncing


        # Automatic chunking detection
        self.chunk_size = chunk_size
        self.use_chunked = chunk_size is not None

        # No need to filter flags anymore since we're not passing them

        # Always store batch_size for both modes
        self.batch_size = batch_size

        if self.use_chunked:
            # Store chunking-specific attributes first
            self._chunk_size = chunk_size
            self._current_chunk = current_chunk if current_chunk is not None else 0
            self._chunk_sampler = ChunkedSampler(len(dataset), chunk_size)
            self._total_chunks = self._calculate_total_chunks()

            # Store parameters for later use
            self.dataloader_kwargs = dataloader_kwargs

            # Initialize parent with original kwargs
            super().__init__(dataset, batch_size=batch_size, sampler=None, **dataloader_kwargs)

        else:
            # Standard DataLoader mode - use parent constructor normally
            super().__init__(dataset, batch_size=batch_size, **dataloader_kwargs)

    def _calculate_total_chunks(self):
        """Calculate how many chunks needed to cover entire dataset."""
        dataset_size = len(self._original_dataset)
        return (dataset_size + self._chunk_size - 1) // self._chunk_size

    def __iter__(self):
        """Return iterator - either chunked or standard."""
        if not self.use_chunked:
            # Use standard DataLoader behavior
            return super().__iter__()

        # Check if we need to advance chunk based on dataset state
        # This handles the case where PyTorch Lightning reuses dataloaders
        self._sync_chunk_with_dataset()

        # Show which chunk is starting
        partition_name = self._dataloader_type if self._dataloader_type else "unknown"

        # Convert 'eval' to 'val' for backwards compatibility
        if partition_name == "eval":
            partition_name = "val"

        print(f"Starting {partition_name} chunk {self._current_chunk} of {self._total_chunks}")

        self._chunk_sampler.set_chunk(self._current_chunk)

        # Remove shuffle from kwargs since we're using a sampler
        # (sampler and shuffle are mutually exclusive in PyTorch DataLoader)
        chunked_kwargs = self.dataloader_kwargs.copy()
        conflicting_params = ['shuffle', 'shuffle_fn', 'generator']
        for param in conflicting_params:
            chunked_kwargs.pop(param, None)

        # Create a new DataLoader instance for this chunk
        dataloader = DataLoader(
            self._original_dataset,
            batch_size=self.batch_size,
            sampler=self._chunk_sampler,
            **chunked_kwargs
        )

        return iter(dataloader)

    def _sync_chunk_with_dataset(self):
        """Sync this dataloader's chunk with the dataset's current chunk."""
        if not self.use_chunked:
            return

        # Try to get the current chunk from the parent dataset
        # This is needed because PyTorch Lightning may reuse dataloaders across epochs
        try:
            if hasattr(self, '_parent_dataset'):
                dataset = self._parent_dataset
            else:
                return

            # Determine which chunk type this dataloader is for
            # Use the stored dataloader_type from when this dataloader was created
            if self._dataloader_type is not None:
                if self._dataloader_type == 'train' and hasattr(dataset, '_current_train_chunk'):
                    expected_chunk = dataset._current_train_chunk
                elif self._dataloader_type == 'eval' and hasattr(dataset, '_current_val_chunk'):
                    expected_chunk = dataset._current_val_chunk
                else:
                    return
            else:
                # Fallback to dataset attribute (old method)
                if hasattr(dataset, '_current_dataloader_type'):
                    if dataset._current_dataloader_type == 'train' and hasattr(dataset, '_current_train_chunk'):
                        expected_chunk = dataset._current_train_chunk
                    elif dataset._current_dataloader_type == 'eval' and hasattr(dataset, '_current_val_chunk'):
                        expected_chunk = dataset._current_val_chunk
                    else:
                        return
                else:
                    return

            # Update our chunk if it's different
            if self._current_chunk != expected_chunk:
                self._current_chunk = expected_chunk
        except Exception:
            # If anything goes wrong, just continue with current chunk
            pass

    def __len__(self):
        """Return the length of the current iteration."""
        if not self.use_chunked:
            # Use standard DataLoader length
            return super().__len__()

        return len(self._chunk_sampler)

    def set_chunk(self, chunk_idx):
        """Set the current chunk index."""
        if self.use_chunked:
            self._current_chunk = chunk_idx % self._total_chunks

    def get_chunk_info(self) -> dict:
        """
        Get information about the current chunking state.

        Returns:
            Dictionary with chunking information
        """
        if not self.use_chunked:
            return {
                'use_chunked': False,
                'mode': 'standard'
            }

        return {
            'use_chunked': True,
            'current_chunk': self._current_chunk,
            'total_chunks': self._total_chunks,
            'chunk_size': self._chunk_size,
            'dataset_size': len(self._original_dataset),
            'samples_in_current_chunk': len(self._chunk_sampler)
        }


    @property
    def is_chunked(self):
        """Check if this dataloader is using chunked mode."""
        return self.use_chunked
