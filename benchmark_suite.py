"""
Comprehensive Benchmarking Suite for MambaDecoderBlock

This suite compares:
1. MambaDecoderBlock vs Regular Mamba (from mambapy)
2. MambaDecoderBlock vs MLABlock

Metrics evaluated:
- Speed (throughput in tokens/second)
- Latency (time per forward pass)
- Scaling behavior (with sequence length and batch size)
- Memory usage (optional)
"""

import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from dataclasses import dataclass
import gc
import warnings

# Note: This script assumes nn.RMSNorm is available (PyTorch 2.0+)
# If you're using an older PyTorch version, you may need to install a compatibility package
# or use a custom RMSNorm implementation

from mambapy.mamba import Mamba, MambaConfig
from mamba_decoder.mamba_block import MambaDecoderBlock, MLABlock, MambaConfig as MDMambaConfig, MoEConfig

warnings.filterwarnings('ignore')


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    sequence_length: int
    batch_size: int
    latency_ms: float
    throughput_tokens_per_sec: float
    memory_mb: Optional[float] = None
    num_parameters: Optional[int] = None


class TorchTimer:
    """Context manager for timing PyTorch operations."""
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        return False


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def benchmark_model(
    model: nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    num_warmup: int = 5,
    num_runs: int = 20,
    use_cuda: bool = True,
) -> BenchmarkResult:
    """
    Benchmark a model's forward pass.
    
    Args:
        model: The model to benchmark
        model_name: Name of the model for logging
        input_tensor: Input tensor (batch_size, seq_len, dim)
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs
        use_cuda: Whether to use CUDA if available
        
    Returns:
        BenchmarkResult with latency and throughput metrics
    """
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)
    
    batch_size, seq_len, dim = input_tensor.shape
    
    # Handle different forward signatures
    forward_kwargs = {}
    if isinstance(model, MLABlock):
        forward_kwargs = {"start_pos": 0, "mask": None}
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(num_warmup):
            if isinstance(model, MLABlock):
                _ = model(input_tensor, **forward_kwargs)
            else:
                _ = model(input_tensor)
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Benchmark runs
    latencies = []
    for _ in range(num_runs):
        with torch.no_grad():
            with TorchTimer() as timer:
                if isinstance(model, MLABlock):
                    _ = model(input_tensor, **forward_kwargs)
                else:
                    _ = model(input_tensor)
            latencies.append(timer.elapsed * 1000)  # Convert to ms
    
    # Calculate statistics
    latency_ms = np.mean(latencies)
    latency_std = np.std(latencies)
    
    # Throughput: tokens per second
    total_tokens = batch_size * seq_len
    throughput_tokens_per_sec = total_tokens / (latency_ms / 1000)
    
    # Memory usage
    memory_mb = get_memory_usage_mb() if torch.cuda.is_available() else None
    
    # Parameter count
    num_params = count_parameters(model)
    
    print(f"{model_name:30s} | SeqLen: {seq_len:4d} | Batch: {batch_size:2d} | "
          f"Latency: {latency_ms:6.2f}±{latency_std:5.2f} ms | "
          f"Throughput: {throughput_tokens_per_sec:8.2f} tokens/s | "
          f"Params: {num_params/1e6:.2f}M")
    
    return BenchmarkResult(
        model_name=model_name,
        sequence_length=seq_len,
        batch_size=batch_size,
        latency_ms=latency_ms,
        throughput_tokens_per_sec=throughput_tokens_per_sec,
        memory_mb=memory_mb,
        num_parameters=num_params,
    )


def create_regular_mamba(dim: int, d_state: int = 16, d_conv: int = 4, expand_factor: int = 2) -> Mamba:
    """Create a regular Mamba model from mambapy."""
    config = MambaConfig(
        d_model=dim,
        n_layers=1,
        d_state=d_state,
        d_conv=d_conv,
        expand_factor=expand_factor,
    )
    return Mamba(config=config)


def create_mamba_decoder_block(dim: int, d_state: int = 16, d_conv: int = 4, 
                               expand_factor: int = 2, n_experts: int = 6, 
                               n_activated: int = 2) -> MambaDecoderBlock:
    """Create a MambaDecoderBlock."""
    mamba_config = MDMambaConfig(
        d_model=dim,
        n_layers=1,
        d_state=d_state,
        d_conv=d_conv,
        expand_factor=expand_factor,
    )
    moe_config = MoEConfig(
        dim=dim,
        n_experts=n_experts,
        n_activated=n_activated,
        expert_inter_dim=None,
        shared_expert_inter_dim=None,
        use_adaptive_bias=True,
        bias_update_rate=0.01,
    )
    return MambaDecoderBlock(
        dim=dim,
        mamba_config=mamba_config,
        moe_config=moe_config,
    )


def create_mla_block(dim: int, max_seq_len: int = 4096, n_heads: int = 16,
                     n_experts: int = 6, n_activated: int = 2) -> MLABlock:
    """Create an MLABlock."""
    return MLABlock(
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        max_batch_size=8,
        max_seq_len=max_seq_len,
        attn_impl="absorb",
        rope_theta=10000.0,
        mscale=1.0,
        n_experts=n_experts,
        n_activated=n_activated,
        expert_inter_dim=None,
        shared_expert_inter_dim=None,
        use_adaptive_bias=True,
        bias_update_rate=0.01,
    )


def run_scaling_benchmarks(
    dim: int = 512,
    sequence_lengths: List[int] = None,
    batch_sizes: List[int] = None,
    use_cuda: bool = True,
) -> List[BenchmarkResult]:
    """
    Run comprehensive scaling benchmarks.
    
    Args:
        dim: Model dimension
        sequence_lengths: List of sequence lengths to test
        batch_sizes: List of batch sizes to test
        use_cuda: Whether to use CUDA
        
    Returns:
        List of BenchmarkResult objects
    """
    if sequence_lengths is None:
        sequence_lengths = [128, 256, 512, 1024, 2048]
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]
    
    results = []
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    
    print("=" * 100)
    print("COMPREHENSIVE BENCHMARKING SUITE")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Model Dimension: {dim}")
    print(f"Sequence Lengths: {sequence_lengths}")
    print(f"Batch Sizes: {batch_sizes}")
    print("=" * 100)
    
    # Create models
    print("\nCreating models...")
    regular_mamba = create_regular_mamba(dim)
    mamba_decoder_block = create_mamba_decoder_block(dim)
    mla_block = create_mla_block(dim, max_seq_len=max(sequence_lengths) * 2)
    
    models = {
        "Regular Mamba": regular_mamba,
        "MambaDecoderBlock": mamba_decoder_block,
        "MLABlock": mla_block,
    }
    
    # Run benchmarks
    for batch_size in batch_sizes:
        for seq_len in sequence_lengths:
            # Create input tensor
            input_tensor = torch.randn(batch_size, seq_len, dim)
            
            for model_name, model in models.items():
                try:
                    result = benchmark_model(
                        model=model,
                        model_name=model_name,
                        input_tensor=input_tensor,
                        num_warmup=5,
                        num_runs=20,
                        use_cuda=use_cuda,
                    )
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error benchmarking {model_name} with seq_len={seq_len}, batch={batch_size}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Clean up after each batch/seq_len combination
            del input_tensor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return results


def plot_benchmark_results(results: List[BenchmarkResult], output_dir: str = "benchmark_plots"):
    """
    Create comprehensive visualization plots from benchmark results.
    
    Args:
        results: List of BenchmarkResult objects
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Organize results by model
    model_results: Dict[str, List[BenchmarkResult]] = {}
    for result in results:
        if result.model_name not in model_results:
            model_results[result.model_name] = []
        model_results[result.model_name].append(result)
    
    # Extract unique sequence lengths and batch sizes
    seq_lengths = sorted(set(r.sequence_length for r in results))
    batch_sizes = sorted(set(r.batch_size for r in results))
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig_size = (16, 10)
    
    # 1. Latency vs Sequence Length (for different batch sizes)
    print("\nGenerating latency vs sequence length plots...")
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    fig.suptitle('Latency vs Sequence Length', fontsize=16, fontweight='bold')
    
    for idx, batch_size in enumerate(batch_sizes[:4]):  # Limit to 4 batch sizes
        ax = axes[idx // 2, idx % 2]
        for model_name in model_results.keys():
            model_data = [r for r in model_results[model_name] if r.batch_size == batch_size]
            if not model_data:
                continue
            seq_lens = [r.sequence_length for r in model_data]
            latencies = [r.latency_ms for r in model_data]
            ax.plot(seq_lens, latencies, marker='o', label=model_name, linewidth=2, markersize=8)
        
        ax.set_xlabel('Sequence Length', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title(f'Batch Size = {batch_size}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/latency_vs_seqlen.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Throughput vs Sequence Length
    print("Generating throughput vs sequence length plots...")
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    fig.suptitle('Throughput vs Sequence Length', fontsize=16, fontweight='bold')
    
    for idx, batch_size in enumerate(batch_sizes[:4]):
        ax = axes[idx // 2, idx % 2]
        for model_name in model_results.keys():
            model_data = [r for r in model_results[model_name] if r.batch_size == batch_size]
            if not model_data:
                continue
            seq_lens = [r.sequence_length for r in model_data]
            throughputs = [r.throughput_tokens_per_sec for r in model_data]
            ax.plot(seq_lens, throughputs, marker='o', label=model_name, linewidth=2, markersize=8)
        
        ax.set_xlabel('Sequence Length', fontsize=12)
        ax.set_ylabel('Throughput (tokens/sec)', fontsize=12)
        ax.set_title(f'Batch Size = {batch_size}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/throughput_vs_seqlen.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Latency vs Batch Size (for different sequence lengths)
    print("Generating latency vs batch size plots...")
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    fig.suptitle('Latency vs Batch Size', fontsize=16, fontweight='bold')
    
    for idx, seq_len in enumerate(seq_lengths[:4]):  # Limit to 4 sequence lengths
        ax = axes[idx // 2, idx % 2]
        for model_name in model_results.keys():
            model_data = [r for r in model_results[model_name] if r.sequence_length == seq_len]
            if not model_data:
                continue
            batch_sizes_plot = [r.batch_size for r in model_data]
            latencies = [r.latency_ms for r in model_data]
            ax.plot(batch_sizes_plot, latencies, marker='s', label=model_name, linewidth=2, markersize=8)
        
        ax.set_xlabel('Batch Size', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title(f'Sequence Length = {seq_len}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/latency_vs_batchsize.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Throughput vs Batch Size
    print("Generating throughput vs batch size plots...")
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    fig.suptitle('Throughput vs Batch Size', fontsize=16, fontweight='bold')
    
    for idx, seq_len in enumerate(seq_lengths[:4]):
        ax = axes[idx // 2, idx % 2]
        for model_name in model_results.keys():
            model_data = [r for r in model_results[model_name] if r.sequence_length == seq_len]
            if not model_data:
                continue
            batch_sizes_plot = [r.batch_size for r in model_data]
            throughputs = [r.throughput_tokens_per_sec for r in model_data]
            ax.plot(batch_sizes_plot, throughputs, marker='s', label=model_name, linewidth=2, markersize=8)
        
        ax.set_xlabel('Batch Size', fontsize=12)
        ax.set_ylabel('Throughput (tokens/sec)', fontsize=12)
        ax.set_title(f'Sequence Length = {seq_len}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/throughput_vs_batchsize.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Speedup comparison (relative to Regular Mamba)
    print("Generating speedup comparison plots...")
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    fig.suptitle('Speedup Relative to Regular Mamba', fontsize=16, fontweight='bold')
    
    for idx, batch_size in enumerate(batch_sizes[:4]):
        ax = axes[idx // 2, idx % 2]
        
        # Get baseline (Regular Mamba)
        baseline_data = [r for r in model_results.get("Regular Mamba", []) if r.batch_size == batch_size]
        if not baseline_data:
            continue
        
        baseline_dict = {r.sequence_length: r.latency_ms for r in baseline_data}
        
        for model_name in model_results.keys():
            if model_name == "Regular Mamba":
                continue
            model_data = [r for r in model_results[model_name] if r.batch_size == batch_size]
            if not model_data:
                continue
            
            seq_lens = []
            speedups = []
            for r in model_data:
                if r.sequence_length in baseline_dict:
                    speedup = baseline_dict[r.sequence_length] / r.latency_ms
                    seq_lens.append(r.sequence_length)
                    speedups.append(speedup)
            
            if seq_lens:
                ax.plot(seq_lens, speedups, marker='o', label=model_name, linewidth=2, markersize=8)
        
        ax.axhline(y=1.0, color='r', linestyle='--', label='Baseline (Regular Mamba)', linewidth=2)
        ax.set_xlabel('Sequence Length', fontsize=12)
        ax.set_ylabel('Speedup (x)', fontsize=12)
        ax.set_title(f'Batch Size = {batch_size}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/speedup_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Scaling efficiency (throughput per parameter)
    print("Generating scaling efficiency plots...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for model_name in model_results.keys():
        model_data = model_results[model_name]
        if not model_data or model_data[0].num_parameters is None:
            continue
        
        # Use first result's parameter count (should be same for all)
        num_params = model_data[0].num_parameters
        
        seq_lens = []
        efficiencies = []
        for r in model_data:
            if r.batch_size == batch_sizes[0]:  # Use first batch size
                efficiency = r.throughput_tokens_per_sec / (num_params / 1e6)  # tokens/sec per M params
                seq_lens.append(r.sequence_length)
                efficiencies.append(efficiency)
        
        if seq_lens:
            ax.plot(seq_lens, efficiencies, marker='o', label=f'{model_name} ({num_params/1e6:.1f}M params)', 
                   linewidth=2, markersize=8)
    
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Throughput per Million Parameters (tokens/sec/M)', fontsize=12)
    ax.set_title('Scaling Efficiency: Throughput per Parameter', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scaling_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Comprehensive comparison heatmap
    print("Generating comprehensive comparison heatmap...")
    seq_len_list = sorted(set(r.sequence_length for r in results))
    batch_size_list = sorted(set(r.batch_size for r in results))
    
    # Create separate heatmaps for each model
    model_names = list(model_results.keys())
    n_models = len(model_names)
    
    fig, axes = plt.subplots(n_models, 2, figsize=(16, 4 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('Comprehensive Comparison Heatmaps', fontsize=16, fontweight='bold')
    
    for model_idx, model_name in enumerate(model_names):
        # Latency heatmap
        ax_latency = axes[model_idx, 0]
        latency_data = np.zeros((len(batch_size_list), len(seq_len_list)))
        
        for r in model_results[model_name]:
            batch_idx = batch_size_list.index(r.batch_size)
            seq_idx = seq_len_list.index(r.sequence_length)
            latency_data[batch_idx, seq_idx] = r.latency_ms
        
        im1 = ax_latency.imshow(latency_data, cmap='viridis', aspect='auto')
        ax_latency.set_xticks(range(len(seq_len_list)))
        ax_latency.set_xticklabels([str(s) for s in seq_len_list])
        ax_latency.set_yticks(range(len(batch_size_list)))
        ax_latency.set_yticklabels([str(b) for b in batch_size_list])
        ax_latency.set_xlabel('Sequence Length', fontsize=11)
        ax_latency.set_ylabel('Batch Size', fontsize=11)
        ax_latency.set_title(f'{model_name} - Latency (ms)', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=ax_latency)
        
        # Throughput heatmap
        ax_throughput = axes[model_idx, 1]
        throughput_data = np.zeros((len(batch_size_list), len(seq_len_list)))
        
        for r in model_results[model_name]:
            batch_idx = batch_size_list.index(r.batch_size)
            seq_idx = seq_len_list.index(r.sequence_length)
            throughput_data[batch_idx, seq_idx] = r.throughput_tokens_per_sec
        
        im2 = ax_throughput.imshow(throughput_data, cmap='plasma', aspect='auto')
        ax_throughput.set_xticks(range(len(seq_len_list)))
        ax_throughput.set_xticklabels([str(s) for s in seq_len_list])
        ax_throughput.set_yticks(range(len(batch_size_list)))
        ax_throughput.set_yticklabels([str(b) for b in batch_size_list])
        ax_throughput.set_xlabel('Sequence Length', fontsize=11)
        ax_throughput.set_ylabel('Batch Size', fontsize=11)
        ax_throughput.set_title(f'{model_name} - Throughput (tokens/sec)', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=ax_throughput)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comprehensive_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Speedup heatmap
    print("Generating speedup heatmap...")
    baseline_data = model_results.get("Regular Mamba", [])
    if baseline_data:
        baseline_dict = {(r.batch_size, r.sequence_length): r.latency_ms for r in baseline_data}
        
        fig, axes = plt.subplots(1, len(model_names) - 1, figsize=(6 * (len(model_names) - 1), 6))
        if len(model_names) - 1 == 1:
            axes = [axes]
        fig.suptitle('Speedup Relative to Regular Mamba', fontsize=16, fontweight='bold')
        
        speedup_idx = 0
        for model_name in model_names:
            if model_name == "Regular Mamba":
                continue
            
            ax = axes[speedup_idx]
            speedup_data = np.zeros((len(batch_size_list), len(seq_len_list)))
            
            for r in model_results[model_name]:
                key = (r.batch_size, r.sequence_length)
                if key in baseline_dict:
                    speedup = baseline_dict[key] / r.latency_ms
                    batch_idx = batch_size_list.index(r.batch_size)
                    seq_idx = seq_len_list.index(r.sequence_length)
                    speedup_data[batch_idx, seq_idx] = speedup
            
            im = ax.imshow(speedup_data, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=2.0)
            ax.set_xticks(range(len(seq_len_list)))
            ax.set_xticklabels([str(s) for s in seq_len_list])
            ax.set_yticks(range(len(batch_size_list)))
            ax.set_yticklabels([str(b) for b in batch_size_list])
            ax.set_xlabel('Sequence Length', fontsize=11)
            ax.set_ylabel('Batch Size', fontsize=11)
            ax.set_title(f'{model_name} Speedup', fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax)
            speedup_idx += 1
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/speedup_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nAll plots saved to {output_dir}/")


def print_summary_table(results: List[BenchmarkResult]):
    """Print a summary table of benchmark results."""
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    
    # Group by model
    model_results: Dict[str, List[BenchmarkResult]] = {}
    for result in results:
        if result.model_name not in model_results:
            model_results[result.model_name] = []
        model_results[result.model_name].append(result)
    
    # Print summary for each model
    for model_name, model_data in model_results.items():
        print(f"\n{model_name}:")
        print("-" * 100)
        
        avg_latency = np.mean([r.latency_ms for r in model_data])
        avg_throughput = np.mean([r.throughput_tokens_per_sec for r in model_data])
        total_params = model_data[0].num_parameters if model_data[0].num_parameters else 0
        
        print(f"  Average Latency: {avg_latency:.2f} ms")
        print(f"  Average Throughput: {avg_throughput:.2f} tokens/sec")
        print(f"  Total Parameters: {total_params/1e6:.2f}M")
        print(f"  Throughput per Million Params: {avg_throughput/(total_params/1e6):.2f} tokens/sec/M")
    
    # Comparison
    print("\n" + "=" * 100)
    print("COMPARISON (Relative to Regular Mamba)")
    print("=" * 100)
    
    baseline = model_results.get("Regular Mamba", [])
    if baseline:
        baseline_avg_latency = np.mean([r.latency_ms for r in baseline])
        baseline_avg_throughput = np.mean([r.throughput_tokens_per_sec for r in baseline])
        
        for model_name, model_data in model_results.items():
            if model_name == "Regular Mamba":
                continue
            
            avg_latency = np.mean([r.latency_ms for r in model_data])
            avg_throughput = np.mean([r.throughput_tokens_per_sec for r in model_data])
            
            latency_speedup = baseline_avg_latency / avg_latency
            throughput_speedup = avg_throughput / baseline_avg_throughput
            
            print(f"\n{model_name}:")
            print(f"  Latency Speedup: {latency_speedup:.2f}x")
            print(f"  Throughput Speedup: {throughput_speedup:.2f}x")


def main():
    """Main benchmarking function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Mamba Benchmarking Suite')
    parser.add_argument('--dim', type=int, default=512, help='Model dimension')
    parser.add_argument('--seq-lens', type=int, nargs='+', default=[128, 256, 512, 1024, 2048],
                       help='Sequence lengths to test')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8],
                       help='Batch sizes to test')
    parser.add_argument('--output-dir', type=str, default='benchmark_plots',
                       help='Output directory for plots')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Run benchmarks
    results = run_scaling_benchmarks(
        dim=args.dim,
        sequence_lengths=args.seq_lens,
        batch_sizes=args.batch_sizes,
        use_cuda=not args.no_cuda,
    )
    
    # Print summary
    print_summary_table(results)
    
    # Generate plots
    plot_benchmark_results(results, output_dir=args.output_dir)
    
    print("\n" + "=" * 100)
    print("BENCHMARKING COMPLETE!")
    print("=" * 100)


if __name__ == "__main__":
    main()

