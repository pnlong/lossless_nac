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

            # Filter out sampler from kwargs to avoid conflicts
            filtered_kwargs = {k: v for k, v in dataloader_kwargs.items() if k != 'sampler'}
            
            # Check if drop_last is True and adjust chunk size accordingly
            drop_last = filtered_kwargs.get('drop_last', True)  # Default to True as per config
            if drop_last:
                # When drop_last=True, we need to ensure chunk_size is divisible by batch_size
                # to get exactly the expected number of batches
                adjusted_chunk_size = (chunk_size // batch_size) * batch_size
                if adjusted_chunk_size != chunk_size:
                    # print(f"🔧 Adjusting chunk size from {chunk_size} to {adjusted_chunk_size} for drop_last=True")
                    # Update the chunk sampler with adjusted size
                    self._chunk_sampler = ChunkedSampler(len(dataset), adjusted_chunk_size)
                    self._chunk_size = adjusted_chunk_size

            # Initialize parent with filtered kwargs
            super().__init__(dataset, batch_size=batch_size, sampler=None, **filtered_kwargs)
            
            # Store original len method
            self._original_len = super().__len__
            
            # Also try overriding other methods PyTorch Lightning might use
            self._original_num_workers = getattr(self, 'num_workers', None)
            self._original_batch_size = getattr(self, 'batch_size', None)

        else:
            # Standard DataLoader mode - filter out sampler to avoid conflicts
            filtered_kwargs = {k: v for k, v in dataloader_kwargs.items() if k != 'sampler'}
            super().__init__(dataset, batch_size=batch_size, **filtered_kwargs)

    def _calculate_total_chunks(self):
        """Calculate how many chunks needed to cover entire dataset."""
        dataset_size = len(self._original_dataset)
        return (dataset_size + self._chunk_size - 1) // self._chunk_size

    def _chunked_len(self):
        """Calculate the correct number of batches for the current chunk."""
        if not self.use_chunked:
            return self._original_len()
        
        # Calculate number of batches in current chunk
        samples_in_chunk = min(self._chunk_size, len(self._original_dataset))
        batches_in_chunk = (samples_in_chunk + self.batch_size - 1) // self.batch_size
        
        # Debug: Print when _chunked_len is called
        # print(f"🔍 ChunkedDataLoader._chunked_len() called: returning {batches_in_chunk} batches")
        # print(f"   Current chunk: {getattr(self, '_current_chunk', 'NOT SET')}, samples_in_chunk: {samples_in_chunk}, batch_size: {self.batch_size}")
        
        return batches_in_chunk
    
    def __len__(self):
        """Override __len__ to ensure PyTorch Lightning calls our method."""
        # print(f"🔍 ChunkedDataLoader.__len__() method called directly")
        if not self.use_chunked:
            return super().__len__()
        
        # Calculate number of batches in current chunk
        samples_in_chunk = min(self._chunk_size, len(self._original_dataset))
        
        # Check if drop_last is True to determine batch calculation
        drop_last = getattr(self, 'drop_last', True)  # Default to True as per config
        if drop_last:
            # When drop_last=True, we get exactly floor(samples_in_chunk / batch_size) batches
            batches_in_chunk = samples_in_chunk // self.batch_size
        else:
            # When drop_last=False, we get ceil(samples_in_chunk / batch_size) batches
            batches_in_chunk = (samples_in_chunk + self.batch_size - 1) // self.batch_size
        
        # print(f"   Returning {batches_in_chunk} batches for chunk with {samples_in_chunk} samples")
        return batches_in_chunk
    
    def __getattr__(self, name):
        """Override __getattr__ to intercept any attribute access."""
        if name == '__len__':
            # print(f"🔍 ChunkedDataLoader.__getattr__('__len__') called")
            return self.__len__
        elif name == 'num_batches':
            # print(f"🔍 ChunkedDataLoader.__getattr__('num_batches') called")
            return self.__len__()
        elif name == 'len':
            # print(f"🔍 ChunkedDataLoader.__getattr__('len') called")
            return self.__len__
        return super().__getattr__(name)

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

        # Calculate batches in this chunk for progress bar
        samples_in_chunk = min(self._chunk_size, len(self._original_dataset))
        batches_in_chunk = (samples_in_chunk + self.batch_size - 1) // self.batch_size
        
        print(f"Starting {partition_name} chunk {self._current_chunk + 1} of {self._total_chunks}")
        # print(f"  Chunk info: {samples_in_chunk} samples, {batches_in_chunk} batches (batch_size={self.batch_size})")

        self._chunk_sampler.set_chunk(self._current_chunk)

        # Remove conflicting parameters from kwargs since we're using a sampler
        # (sampler and shuffle are mutually exclusive in PyTorch DataLoader)
        chunked_kwargs = self.dataloader_kwargs.copy()
        conflicting_params = ['shuffle', 'shuffle_fn', 'generator', 'sampler']
        for param in conflicting_params:
            chunked_kwargs.pop(param, None)

        # Create a new DataLoader instance for this chunk
        dataloader = DataLoader(
            self._original_dataset,
            batch_size=self.batch_size,
            sampler=self._chunk_sampler,
            **chunked_kwargs
        )
        
        # Override the internal dataloader's __len__ method to return correct batch count
        def chunked_len():
            samples_in_chunk = min(self._chunk_size, len(self._original_dataset))
            batches_in_chunk = (samples_in_chunk + self.batch_size - 1) // self.batch_size
            # print(f"🔍 Internal DataLoader.__len__() called: returning {batches_in_chunk} batches")
            # print(f"   samples_in_chunk: {samples_in_chunk}, batch_size: {self.batch_size}")
            return batches_in_chunk
        
        dataloader.__len__ = chunked_len
        
        # Also override num_batches property if it exists (some PyTorch versions use this)
        if hasattr(dataloader, 'num_batches'):
            dataloader.num_batches = chunked_len()

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
                elif self._dataloader_type in ['eval', 'val'] and hasattr(dataset, '_current_val_chunk'):
                    expected_chunk = dataset._current_val_chunk
                elif self._dataloader_type == 'test' and hasattr(dataset, '_current_test_chunk'):
                    expected_chunk = dataset._current_test_chunk
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
