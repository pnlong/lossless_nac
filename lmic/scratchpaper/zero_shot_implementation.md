# Zero-Shot Evaluation Script Implementation Plan

## Overview

This document outlines the implementation plan for `zero_shot.py`, a comprehensive script that evaluates different pre-trained language models on custom audio data without training. The script will provide extensive configurability for audio processing parameters and support multiple evaluation scenarios.

## Script Architecture

### Core Design Principles

1. **Modular Design**: Separate concerns into distinct functions for model loading, data processing, evaluation, and result reporting
2. **Flexible Configuration**: Support extensive command-line parameterization for all audio processing options
3. **Multiple Model Support**: Handle different pre-trained models with automatic configuration detection
4. **Comprehensive Evaluation**: Support both single-model and comparative evaluations
5. **Robust Error Handling**: Graceful handling of missing files, invalid configurations, and processing errors

### Main Components

```
zero_shot.py
├── Command-line argument parsing
├── Model management system
├── Audio data processing pipeline
├── Evaluation engine
├── Result reporting and logging
└── Utility functions
```

## Command-Line Interface Design

### Required Arguments

- `--audio_dir`: Path to directory containing WAV files (required)
- `--model_path`: Path to pre-trained model file (params.npz) (required)

### Audio Processing Parameters

- `--use_16bit`: Use 16-bit audio processing (vocab size 65536) vs 8-bit (vocab size 256)
- `--stereo_blocking_n`: Size of blocks for stereo processing in samples (default: 1024)
- `--chunk_size`: Size of each compression chunk in bytes (default: 2048)
- `--num_chunks`: Maximum number of chunks to evaluate (default: 1000)

### Model Configuration Parameters

- `--embedding_dim`: Embedding dimension for model (default: 64)
- `--num_layers`: Number of transformer layers (default: 4)
- `--num_heads`: Number of attention heads (default: 8)
- `--widening_factor`: Feedforward network scaling factor (default: 4)

### Evaluation Parameters

- `--compressor`: Compressor to use for comparison (default: language_model)
- `--baseline_compressors`: List of baseline compressors to compare against (default: ['gzip', 'flac', 'lzma'])
- `--output_file`: Path to save detailed results (optional)
- `--verbose`: Enable verbose logging (default: False)
- `--use_tqdm`: Show progress bars (default: True)

### Advanced Options

- `--slow_compression`: Use slow lossless compression mode (default: False)
- `--seed`: Random seed for reproducibility (default: 42)
- `--gpu`: Use GPU acceleration if available (default: True)

## Implementation Structure

### 1. Argument Parsing and Validation

```python
def parse_arguments():
    """Parse and validate command-line arguments."""
    # Use argparse with comprehensive argument definitions
    # Include validation functions for each parameter type
    # Provide helpful error messages and usage examples
```

**Key Features:**
- Comprehensive argument validation
- Automatic type conversion and range checking
- Helpful error messages with suggested fixes
- Support for configuration files (future enhancement)

### 2. Model Management System

```python
def load_model_parameters(model_path: str) -> hk.Params:
    """Load pre-trained model parameters from file."""
    
def detect_model_config(model_path: str) -> dict:
    """Detect model configuration from parameter file."""
    
def create_model_predict_fn(params: hk.Params, config: dict) -> Callable:
    """Create prediction function for loaded model."""
```

**Key Features:**
- Support for multiple model file formats (.npz, .pkl, etc.)
- Automatic model configuration detection
- Graceful handling of missing or corrupted model files
- Model compatibility validation

### 3. Audio Data Processing Pipeline

```python
def setup_audio_data_generator(args) -> Iterator[bytes]:
    """Set up audio data generator with specified parameters."""
    
def validate_audio_configuration(args) -> None:
    """Validate audio processing configuration."""
    
def get_audio_file_paths(audio_dir: str) -> List[str]:
    """Discover and validate audio files in directory."""
```

**Key Features:**
- Integration with existing `get_custom_audio_iterator` function
- Comprehensive parameter validation
- Support for both mono and stereo audio files
- Automatic format detection and handling

### 4. Evaluation Engine

```python
def evaluate_single_model(model_path: str, data_generator: Iterator[bytes], args) -> dict:
    """Evaluate a single model on the provided data."""
    
def evaluate_baseline_compressors(data_generator: Iterator[bytes], compressors: List[str]) -> dict:
    """Evaluate baseline compressors for comparison."""
    
def run_comprehensive_evaluation(args) -> dict:
    """Run complete evaluation with all specified models and baselines."""
```

**Key Features:**
- Support for both single-model and comparative evaluations
- Integration with existing `evaluate_compressor_chunked` function
- Comprehensive timing and compression ratio metrics
- Error handling and recovery for individual evaluation failures

### 5. Result Reporting and Logging

```python
def format_results(results: dict, args) -> str:
    """Format evaluation results for display."""
    
def save_results(results: dict, output_file: str) -> None:
    """Save detailed results to file."""
    
def print_summary(results: dict) -> None:
    """Print summary of evaluation results."""
```

**Key Features:**
- Multiple output formats (console, JSON, CSV)
- Detailed compression statistics and timing information
- Comparison tables for multiple models/compressors
- Configurable logging levels and output destinations

## Detailed Implementation Plan

### Phase 1: Core Infrastructure

1. **Set up project structure**
   - Create main `zero_shot.py` file
   - Import necessary dependencies
   - Set up logging configuration

2. **Implement argument parsing**
   - Define all command-line arguments
   - Add validation functions
   - Create help text and usage examples

3. **Create model management functions**
   - Implement model loading with error handling
   - Add model configuration detection
   - Create prediction function factory

### Phase 2: Audio Processing Integration

1. **Integrate with existing audio pipeline**
   - Connect to `get_custom_audio_iterator` function
   - Add parameter validation and conversion
   - Implement audio file discovery

2. **Add configuration validation**
   - Validate audio processing parameters
   - Check file existence and format
   - Provide helpful error messages

### Phase 3: Evaluation Engine

1. **Implement single-model evaluation**
   - Create evaluation wrapper functions
   - Add timing and compression ratio calculation
   - Implement error handling and recovery

2. **Add baseline compressor support**
   - Integrate with existing compressor framework
   - Support multiple baseline comparisons
   - Add comparative analysis functions

### Phase 4: Result Reporting

1. **Create result formatting functions**
   - Design output format for console display
   - Add JSON/CSV export capabilities
   - Create comparison tables

2. **Implement logging and progress tracking**
   - Add configurable logging levels
   - Implement progress bars for long evaluations
   - Add detailed error reporting

### Phase 5: Advanced Features

1. **Add batch evaluation support**
   - Support multiple model evaluation in single run
   - Add parallel processing capabilities
   - Implement result aggregation

2. **Enhance error handling and recovery**
   - Add retry mechanisms for failed evaluations
   - Implement partial result saving
   - Add comprehensive error reporting

## Usage Examples

### Basic Usage
```bash
python zero_shot.py --audio_dir /path/to/audio --model_path /path/to/model.npz
```

### Advanced Configuration
```bash
python zero_shot.py \
    --audio_dir /path/to/audio \
    --model_path /path/to/model.npz \
    --use_16bit \
    --stereo_blocking_n 2048 \
    --chunk_size 4096 \
    --num_chunks 5000 \
    --baseline_compressors gzip flac lzma \
    --output_file results.json \
    --verbose
```

### Comparative Evaluation
```bash
python zero_shot.py \
    --audio_dir /path/to/audio \
    --model_path /path/to/model1.npz \
    --baseline_compressors gzip flac \
    --embedding_dim 128 \
    --num_layers 6 \
    --use_16bit
```

## Expected Output Format

### Console Output
```
Zero-Shot Language Model Evaluation Results
==========================================

Configuration:
  Audio Directory: /path/to/audio
  Model: /path/to/model.npz
  16-bit Audio: True
  Stereo Blocking: 2048 samples
  Chunk Size: 4096 bytes
  Number of Chunks: 5000

Results:
  Language Model: 15.2% compression ratio (2.3s)
  FLAC Baseline: 28.7% compression ratio (0.8s)
  GZIP Baseline: 45.1% compression ratio (1.2s)
  LZMA Baseline: 38.9% compression ratio (2.1s)

Performance Summary:
  Best Compressor: Language Model (15.2%)
  Improvement over FLAC: 47.0% better
  Total Evaluation Time: 6.4s
```

### JSON Output
```json
{
  "configuration": {
    "audio_directory": "/path/to/audio",
    "model_path": "/path/to/model.npz",
    "use_16bit": true,
    "stereo_blocking_n": 2048,
    "chunk_size": 4096,
    "num_chunks": 5000
  },
  "results": {
    "language_model": {
      "compression_ratio": 0.152,
      "compression_time": 2.3,
      "original_size": 20480000,
      "compressed_size": 3112960
    },
    "flac": {
      "compression_ratio": 0.287,
      "compression_time": 0.8,
      "original_size": 20480000,
      "compressed_size": 5877760
    }
  },
  "summary": {
    "best_compressor": "language_model",
    "total_evaluation_time": 6.4,
    "improvements": {
      "vs_flac": 0.47,
      "vs_gzip": 0.66,
      "vs_lzma": 0.61
    }
  }
}
```

## Error Handling Strategy

### Common Error Scenarios

1. **Missing Model File**
   - Check file existence before loading
   - Provide helpful error message with suggested actions
   - Support for multiple model file formats

2. **Invalid Audio Configuration**
   - Validate all audio processing parameters
   - Check parameter compatibility (e.g., 16-bit with even chunk sizes)
   - Provide specific error messages with suggested fixes

3. **Audio File Issues**
   - Validate audio file existence and format
   - Handle corrupted or unsupported audio files
   - Provide detailed error reporting for problematic files

4. **Memory and Performance Issues**
   - Monitor memory usage during evaluation
   - Provide warnings for large datasets
   - Support for chunked processing to handle memory constraints

### Recovery Mechanisms

1. **Partial Result Saving**
   - Save results incrementally during long evaluations
   - Allow resumption of interrupted evaluations
   - Provide status reporting for long-running evaluations

2. **Graceful Degradation**
   - Continue evaluation even if some compressors fail
   - Provide partial results with error indicators
   - Log detailed error information for debugging

## Testing Strategy

### Unit Tests

1. **Argument parsing and validation**
2. **Model loading and configuration detection**
3. **Audio data processing pipeline**
4. **Result formatting and reporting**

### Integration Tests

1. **End-to-end evaluation workflows**
2. **Multiple model comparison scenarios**
3. **Error handling and recovery mechanisms**
4. **Performance and memory usage validation**

### Test Data

1. **Small audio dataset for quick testing**
2. **Various audio formats and configurations**
3. **Pre-trained model files for testing**
4. **Edge cases and error scenarios**

## Future Enhancements

### Phase 6: Advanced Features

1. **Configuration File Support**
   - YAML/JSON configuration files
   - Template configurations for common scenarios
   - Configuration validation and inheritance

2. **Parallel Processing**
   - Multi-threaded evaluation for multiple models
   - GPU acceleration support
   - Distributed evaluation capabilities

3. **Advanced Analytics**
   - Compression ratio analysis across different audio types
   - Performance profiling and optimization suggestions
   - Statistical analysis of compression patterns

4. **Integration with Training Pipeline**
   - Automatic model evaluation after training
   - Integration with wandb for experiment tracking
   - Continuous evaluation during training

## Dependencies and Requirements

### Core Dependencies
- `argparse`: Command-line argument parsing
- `logging`: Comprehensive logging system
- `json`: Result serialization
- `pathlib`: File path handling
- `typing`: Type hints and annotations

### Framework Dependencies
- `language_modeling_is_compression`: Core framework
- `numpy`: Numerical operations
- `jax`: Model inference
- `haiku`: Model parameter handling
- `tqdm`: Progress bars

### Optional Dependencies
- `wandb`: Experiment tracking (optional)
- `matplotlib`: Result visualization (future)
- `pandas`: Data analysis (future)

## Implementation Timeline

### Week 1: Core Infrastructure
- Set up project structure and dependencies
- Implement argument parsing and validation
- Create basic model loading functionality

### Week 2: Audio Processing Integration
- Integrate with existing audio pipeline
- Add comprehensive parameter validation
- Implement audio file discovery and validation

### Week 3: Evaluation Engine
- Implement single-model evaluation
- Add baseline compressor support
- Create comprehensive evaluation workflows

### Week 4: Result Reporting and Testing
- Implement result formatting and export
- Add comprehensive logging and error handling
- Create unit and integration tests

### Week 5: Polish and Documentation
- Add advanced features and optimizations
- Create comprehensive documentation
- Performance testing and optimization

This implementation plan provides a comprehensive roadmap for creating a robust, flexible, and user-friendly zero-shot evaluation script that integrates seamlessly with the existing language_modeling_is_compression framework.
