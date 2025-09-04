# Chunked Training Debug Guide

## 🎯 Quick Verification: Is Chunked Training Working?

### Method 1: Environment Variables (Recommended)
```bash
# Enable debugging for both components
export CHUNKED_SAMPLER_DEBUG=true
export CHUNKED_DATALOADER_DEBUG=true

# Run your training
python train.py ...
```

### Method 2: Code-based Debugging
```python
from src.dataloaders.chunked_sampler import ChunkedSampler
from src.dataloaders.chunked_dataloader import ChunkedDataLoader

# Enable debug in your dataset class
def train_dataloader(self, **kwargs):
    return ChunkedDataLoader(
        self.dataset_train,
        batch_size=self.batch_size,
        chunk_size=self.train_chunk_size,
        debug=True,  # Enable debug output
        **kwargs
    )
```

## 🔍 Debug Output You'll See

### During Training Initialization:
```
🔧 ChunkedSampler initialized: dataset_size=1000, chunk_size=200, total_chunks=5
🔧 ChunkedDataLoader initialized: chunk_size=200, total_chunks=5
```

### During Each Training Epoch:
```
🔄 ChunkedDataLoader.__iter__: Starting chunk 0 of 5
🔄 Chunk 0: NO_WRAP - start=0, end=200
📊 Chunk 0: Generated 200 indices, range=[0, 199]
⏭️  ChunkedDataLoader.__iter__: Advanced from chunk 0 to 1

🔄 ChunkedDataLoader.__iter__: Starting chunk 1 of 5
🔄 Chunk 1: NO_WRAP - start=200, end=400
📊 Chunk 1: Generated 200 indices, range=[200, 399]
⏭️  ChunkedDataLoader.__iter__: Advanced from chunk 1 to 2
```

### During Validation:
```
🔄 ChunkedDataLoader.__iter__: Starting chunk 0 of 5
🔄 Chunk 0: NO_WRAP - start=0, end=200
📊 Chunk 0: Generated 200 indices, range=[0, 199]
⏭️  ChunkedDataLoader.__iter__: Advanced from chunk 0 to 1
```

## ✅ Verification Checklist

When chunked training is working correctly, you should see:

1. **🔢 Different chunk numbers** progressing (0→1→2→3→4→0...)
2. **📊 Different index ranges** for each chunk:
   - Chunk 0: [0, 199]
   - Chunk 1: [200, 399]
   - Chunk 2: [400, 599]
   - etc.
3. **🔄 Wrapping behavior** when reaching dataset end
4. **⏭️ Chunk advancement** after each iteration

## 🚨 Common Issues

### Same chunk repeated:
```
🔄 ChunkedDataLoader.__iter__: Starting chunk 0 of 5
🔄 ChunkedDataLoader.__iter__: Starting chunk 0 of 5  # ← Same chunk!
```
**Cause**: DataLoader iterator consumed multiple times without advancing chunk.

### Overlapping indices:
```
Chunk 0: indices [0, 199]
Chunk 1: indices [50, 249]  # ← Overlap with chunk 0!
```
**Cause**: Chunk size calculation error or wrapping logic issue.

## 🎛️ Configuration

### Enable in YAML:
```yaml
train:
  train_chunk_size: 2000  # Enable chunked training
  val_chunk_size: 200     # Enable chunked validation
```

### Disable Debugging:
```yaml
# Remove environment variables
unset CHUNKED_SAMPLER_DEBUG
unset CHUNKED_DATALOADER_DEBUG

# Or set to false
export CHUNKED_SAMPLER_DEBUG=false
export CHUNKED_DATALOADER_DEBUG=false
```

## 🔧 Troubleshooting

### No debug output:
- Check environment variables are set: `echo $CHUNKED_SAMPLER_DEBUG`
- Ensure chunk_size parameters are specified in config
- Verify you're using ChunkedDataLoader, not standard DataLoader

### Performance impact:
- Debug mode adds minimal overhead (~1-2% slower)
- Safe to leave enabled during development
- Disable for production training

## 📈 Expected Behavior

**With chunking enabled:**
- Each epoch processes the entire dataset in chunks
- Training batches come from different dataset segments each epoch
- Validation batches follow the same chunking pattern
- Debug output shows chunk progression and index ranges

**Without chunking (standard):**
- No debug output from chunked components
- Standard PyTorch DataLoader behavior
- Same data patterns each epoch (if no shuffling)
