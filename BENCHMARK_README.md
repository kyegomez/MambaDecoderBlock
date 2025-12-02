# Comprehensive Benchmarking Suite

This benchmarking suite provides extensive performance comparisons between:
1. **MambaDecoderBlock** vs **Regular Mamba** (from mambapy)
2. **MambaDecoderBlock** vs **MLABlock** (Multi-head Latent Attention)

## Metrics Evaluated

- **Speed (Throughput)**: Tokens processed per second
- **Latency**: Time per forward pass in milliseconds
- **Scaling Behavior**: Performance across different sequence lengths and batch sizes
- **Parameter Efficiency**: Throughput per million parameters

## Quick Start

### Basic Usage

```bash
# Run with default settings
python benchmark_suite.py

# Or use the convenience script
python run_benchmark.py
```

### Custom Configuration

```bash
# Custom model dimension and test ranges
python run_benchmark.py --dim 1024 --seq-lens 256 512 1024 2048 --batch-sizes 1 2 4 8

# CPU-only mode
python run_benchmark.py --no-cuda

# Custom output directory
python run_benchmark.py --output-dir my_benchmark_results
```

## Command Line Arguments

- `--dim`: Model dimension (default: 512)
- `--seq-lens`: Sequence lengths to test (default: [128, 256, 512, 1024, 2048])
- `--batch-sizes`: Batch sizes to test (default: [1, 2, 4, 8])
- `--output-dir`: Directory to save plots (default: 'benchmark_plots')
- `--no-cuda`: Disable CUDA (use CPU only)

## Output

The benchmark suite generates:

1. **Latency vs Sequence Length** plots (for different batch sizes)
2. **Throughput vs Sequence Length** plots (for different batch sizes)
3. **Latency vs Batch Size** plots (for different sequence lengths)
4. **Throughput vs Batch Size** plots (for different sequence lengths)
5. **Speedup Comparison** plots (relative to Regular Mamba)
6. **Scaling Efficiency** plots (throughput per parameter)
7. **Comprehensive Heatmaps** (latency and throughput for all configurations)
8. **Speedup Heatmaps** (relative performance gains)

All plots are saved as high-resolution PNG files (300 DPI) in the specified output directory.

## Example Output

```
====================================================================================================
COMPREHENSIVE BENCHMARKING SUITE
====================================================================================================
Device: cuda
Model Dimension: 512
Sequence Lengths: [128, 256, 512, 1024, 2048]
Batch Sizes: [1, 2, 4, 8]
====================================================================================================

Regular Mamba              | SeqLen:  128 | Batch:  1 | Latency:  12.34±0.56 ms | Throughput:  10368.42 tokens/s | Params: 2.45M
MambaDecoderBlock          | SeqLen:  128 | Batch:  1 | Latency:  15.67±0.78 ms | Throughput:   8168.23 tokens/s | Params: 3.21M
MLABlock                   | SeqLen:  128 | Batch:  1 | Latency:  18.92±1.12 ms | Throughput:   6764.89 tokens/s | Params: 4.12M
...
```

## Programmatic Usage

You can also use the benchmark suite programmatically:

```python
from benchmark_suite import run_scaling_benchmarks, plot_benchmark_results, print_summary_table

# Run benchmarks
results = run_scaling_benchmarks(
    dim=512,
    sequence_lengths=[128, 256, 512, 1024],
    batch_sizes=[1, 2, 4],
    use_cuda=True,
)

# Print summary
print_summary_table(results)

# Generate plots
plot_benchmark_results(results, output_dir='my_results')
```

## Benchmark Details

### Models Compared

1. **Regular Mamba**: Standard Mamba implementation from mambapy
   - Single Mamba layer
   - Baseline for comparison

2. **MambaDecoderBlock**: Mamba + MoE decoder block
   - Mamba state space model
   - Mixture of Experts (MoE) feed-forward
   - RMSNorm normalization

3. **MLABlock**: Multi-head Latent Attention + MoE block
   - SimpleMLA attention mechanism
   - LoRA-based projections
   - Rotary positional embeddings (RoPE)
   - Mixture of Experts (MoE) feed-forward

### Benchmark Methodology

- **Warmup Runs**: 5 warmup iterations to stabilize performance
- **Benchmark Runs**: 20 timed iterations per configuration
- **Statistics**: Mean latency with standard deviation
- **Memory**: GPU memory usage (if CUDA available)
- **Parameters**: Total trainable parameters per model

## Requirements

- PyTorch (2.0+ recommended for nn.RMSNorm support)
- matplotlib
- numpy
- mambapy
- open_kimi (for MoE)

## Notes

- The benchmark suite automatically handles CUDA synchronization for accurate timing
- Models are kept in memory during benchmarking for efficiency
- Results include both raw metrics and relative comparisons
- All timing measurements account for GPU synchronization overhead

## Troubleshooting

### RMSNorm Not Available

If you encounter errors about `nn.RMSNorm` not being available, you may need to:
- Upgrade to PyTorch 2.0+
- Or install a compatibility package that provides RMSNorm

### Out of Memory Errors

If you run out of GPU memory:
- Reduce the maximum sequence length
- Reduce batch sizes
- Use `--no-cuda` to run on CPU (slower but more memory available)

### Import Errors

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

