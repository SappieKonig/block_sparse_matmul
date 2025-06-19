import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from test2 import (
    generate_column_block_sparse_matrix, 
    full_tensor, 
    torch_blockwise_matmul, 
    triton_mm, 
    sparse_triton_mm
)


def benchmark_method(func, *args, iters=20, warmup=5):
    """Benchmark a method with proper warmup and timing."""
    # Warmup
    for _ in range(warmup):
        func(*args)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iters):
            result = func(*args)
        end_event.record()
        torch.cuda.synchronize()
        
        return start_event.elapsed_time(end_event) / iters, result
    else:
        start_time = time.perf_counter()
        for _ in range(iters):
            result = func(*args)
        end_time = time.perf_counter()
        
        return ((end_time - start_time) * 1000) / iters, result


def run_comprehensive_benchmark():
    """Run benchmark across different problem sizes."""
    device = "cuda"
    dtype = torch.float16
    block_size = 128
    occupancy = 4
    
    # Different problem sizes to test
    configs = [
        {"num_blocks": 8, "batch_size": 256},
        {"num_blocks": 16, "batch_size": 512}, 
        {"num_blocks": 32, "batch_size": 1024},
        {"num_blocks": 64, "batch_size": 2048},
    ]
    
    results = {
        "Dense PyTorch": [],
        "Blockwise PyTorch": [],
        "Triton Dense": [],
        "Triton Sparse": []
    }
    
    problem_sizes = []
    sparsities = []
    
    print("Running comprehensive benchmark...")
    print("=" * 60)
    
    for config in configs:
        num_blocks = config["num_blocks"]
        batch_size = config["batch_size"]
        
        print(f"\nProblem size: {num_blocks}x{num_blocks} blocks, batch={batch_size}")
        print(f"Matrix size: {num_blocks*block_size}x{num_blocks*block_size}")
        
        # Generate test data
        B = torch.randn((batch_size, num_blocks*block_size), dtype=dtype, device=device)
        data, coordinates = generate_column_block_sparse_matrix(block_size, num_blocks, occupancy, device, dtype)
        A_full = full_tensor(data, coordinates, block_size)
        torch_coords = torch.tensor(coordinates, dtype=torch.int32, device=device)
        
        sparsity = 1 - (occupancy / num_blocks)
        sparsities.append(sparsity)
        problem_sizes.append(f"{num_blocks}x{num_blocks}")
        
        print(f"Sparsity: {sparsity:.1%}")
        
        # 1. Dense PyTorch (reference)
        print("  Testing Dense PyTorch...", end=" ")
        time_dense, result_dense = benchmark_method(lambda: B @ A_full)
        results["Dense PyTorch"].append(time_dense)
        print(f"{time_dense:.2f} ms")
        
        # 2. Blockwise PyTorch
        print("  Testing Blockwise PyTorch...", end=" ")
        time_blockwise, result_blockwise = benchmark_method(torch_blockwise_matmul, data, coordinates, B, block_size)
        results["Blockwise PyTorch"].append(time_blockwise)
        print(f"{time_blockwise:.2f} ms")
        
        # 3. Triton Dense
        print("  Testing Triton Dense...", end=" ")
        time_triton_dense, result_triton_dense = benchmark_method(triton_mm, B, A_full)
        results["Triton Dense"].append(time_triton_dense)
        print(f"{time_triton_dense:.2f} ms")
        
        # 4. Triton Sparse
        print("  Testing Triton Sparse...", end=" ")
        time_triton_sparse, result_triton_sparse = benchmark_method(sparse_triton_mm, B, data, torch_coords)
        results["Triton Sparse"].append(time_triton_sparse)
        print(f"{time_triton_sparse:.2f} ms")
        
        # Verify correctness
        max_error_blockwise = (result_dense.float() - result_blockwise.float()).abs().max().item()
        max_error_triton_dense = (result_dense.float() - result_triton_dense.float()).abs().max().item()
        max_error_triton_sparse = (result_dense.float() - result_triton_sparse.float()).abs().max().item()
        
        print(f"  Max errors: Blockwise={max_error_blockwise:.2e}, "
              f"Triton Dense={max_error_triton_dense:.2e}, "
              f"Triton Sparse={max_error_triton_sparse:.2e}")
    
    return results, problem_sizes, sparsities, configs


def calculate_flops(batch_size, matrix_size, sparsity=0.0):
    """Calculate FLOPS for matrix multiplication."""
    # For B @ A where B is (batch_size, matrix_size) and A is (matrix_size, matrix_size)
    # Dense FLOPS: batch_size * matrix_size * matrix_size * 2 (multiply-add)
    dense_flops = batch_size * matrix_size * matrix_size * 2
    
    # Sparse FLOPS: account for sparsity in matrix A
    sparse_flops = dense_flops * (1 - sparsity)
    
    return dense_flops, sparse_flops


def create_visualization(results, problem_sizes, sparsities, configs):
    """Create and save matplotlib visualization with TFLOPS."""
    
    # Set up the plot style
    plt.style.use('default')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Colors for each method
    colors = {
        "Dense PyTorch": "#1f77b4",
        "Blockwise PyTorch": "#ff7f0e", 
        "Triton Dense": "#2ca02c",
        "Triton Sparse": "#d62728"
    }
    
    # Calculate FLOPS for each configuration
    all_flops = []
    for i, config in enumerate(configs):
        num_blocks = config["num_blocks"]
        batch_size = config["batch_size"]
        matrix_size = num_blocks * 128  # block_size = 128
        sparsity = sparsities[i]
        
        dense_flops, sparse_flops = calculate_flops(batch_size, matrix_size, sparsity)
        all_flops.append((dense_flops, sparse_flops))
    
    # Plot 1: Absolute performance
    x = np.arange(len(problem_sizes))
    width = 0.2
    
    for i, (method, times) in enumerate(results.items()):
        offset = (i - 1.5) * width
        bars = ax1.bar(x + offset, times, width, label=method, color=colors[method], alpha=0.8)
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{time:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Problem Size (blocks)')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Performance Comparison (Time)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(problem_sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Speedup relative to Dense PyTorch
    dense_times = results["Dense PyTorch"]
    
    for method, times in results.items():
        if method != "Dense PyTorch":
            speedups = [dense_times[i] / times[i] for i in range(len(times))]
            ax2.plot(x, speedups, marker='o', linewidth=2, markersize=8, 
                    label=method, color=colors[method])
            
            # Add speedup values as text
            for i, speedup in enumerate(speedups):
                ax2.text(x[i], speedup + 0.05, f'{speedup:.2f}x', 
                        ha='center', va='bottom', fontsize=8)
    
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Dense PyTorch (baseline)')
    ax2.set_xlabel('Problem Size (blocks)')
    ax2.set_ylabel('Speedup (relative to Dense PyTorch)')
    ax2.set_title('Speedup Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(problem_sizes)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # Plot 3: TFLOPS throughput
    for method, times in results.items():
        tflops_values = []
        for i, time_ms in enumerate(times):
            # Choose appropriate FLOPS (dense for dense methods, sparse for sparse methods)
            if "Sparse" in method or "Blockwise" in method:
                flops = all_flops[i][1]  # sparse FLOPS
            else:
                flops = all_flops[i][0]  # dense FLOPS
            
            # Convert to TFLOPS: FLOPS / (time_ms * 1e-3) / 1e12
            tflops = flops / (time_ms * 1e-3) / 1e12
            tflops_values.append(tflops)
        
        ax3.plot(x, tflops_values, marker='o', linewidth=2, markersize=8, 
                label=method, color=colors[method])
        
        # Add TFLOPS annotations on the right
        if len(tflops_values) > 0:
            last_tflops = tflops_values[-1]
            ax3.text(x[-1] + 0.1, last_tflops, f'{last_tflops:.1f}', 
                    va='center', ha='left', fontsize=9, color=colors[method], weight='bold')
    
    ax3.set_xlabel('Problem Size (blocks)')
    ax3.set_ylabel('Throughput (TFLOPS)')
    ax3.set_title('TFLOPS Throughput')
    ax3.set_xticks(x)
    ax3.set_xticklabels(problem_sizes)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(left=-0.5, right=len(x)-0.3)  # Make room for annotations
    
    # Add sparsity information
    sparsity_text = f"Sparsity: {sparsities[0]:.1%} (occupancy={4}/{problem_sizes[0].split('x')[0]} blocks per column)"
    fig.suptitle(f'Column Block Sparse Matrix Multiplication Benchmark\n{sparsity_text}', 
                fontsize=14, y=0.98)
    
    plt.tight_layout()
    
    # Save the figure
    filename = 'matrix_multiplication_benchmark.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {filename}")
    
    return fig


def print_summary_table(results, problem_sizes, sparsities, configs):
    """Print a nicely formatted summary table with TFLOPS."""
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY - TIMING (ms)")
    print("=" * 100)
    
    # Header
    print(f"{'Problem Size':<15} {'Dense PyTorch':<15} {'Blockwise':<15} {'Triton Dense':<15} {'Triton Sparse':<15}")
    print("-" * 100)
    
    # Data rows
    for i, size in enumerate(problem_sizes):
        dense = results["Dense PyTorch"][i]
        blockwise = results["Blockwise PyTorch"][i]
        triton_dense = results["Triton Dense"][i]
        triton_sparse = results["Triton Sparse"][i]
        
        print(f"{size:<15} {dense:<15.2f} {blockwise:<15.2f} {triton_dense:<15.2f} {triton_sparse:<15.2f}")
    
    print("-" * 100)
    
    # TFLOPS summary
    print("\nTFLOPS THROUGHPUT:")
    print("=" * 100)
    print(f"{'Problem Size':<15} {'Dense PyTorch':<15} {'Blockwise':<15} {'Triton Dense':<15} {'Triton Sparse':<15}")
    print("-" * 100)
    
    for i, size in enumerate(problem_sizes):
        config = configs[i]
        num_blocks = config["num_blocks"]
        batch_size = config["batch_size"]
        matrix_size = num_blocks * 128
        sparsity = sparsities[i]
        
        dense_flops, sparse_flops = calculate_flops(batch_size, matrix_size, sparsity)
        
        # Calculate TFLOPS for each method
        dense_tflops = dense_flops / (results["Dense PyTorch"][i] * 1e-3) / 1e12
        blockwise_tflops = sparse_flops / (results["Blockwise PyTorch"][i] * 1e-3) / 1e12
        triton_dense_tflops = dense_flops / (results["Triton Dense"][i] * 1e-3) / 1e12
        triton_sparse_tflops = sparse_flops / (results["Triton Sparse"][i] * 1e-3) / 1e12
        
        print(f"{size:<15} {dense_tflops:<15.1f} {blockwise_tflops:<15.1f} {triton_dense_tflops:<15.1f} {triton_sparse_tflops:<15.1f}")
    
    print("-" * 100)
    
    # Speedup summary
    print("\nSPEEDUP RELATIVE TO DENSE PYTORCH:")
    print("-" * 50)
    
    dense_times = results["Dense PyTorch"]
    for method in ["Blockwise PyTorch", "Triton Dense", "Triton Sparse"]:
        times = results[method]
        avg_speedup = np.mean([dense_times[i] / times[i] for i in range(len(times))])
        print(f"{method:<20}: {avg_speedup:.2f}x average speedup")


if __name__ == "__main__":
    # Run the comprehensive benchmark
    results, problem_sizes, sparsities, configs = run_comprehensive_benchmark()
    
    # Create and save visualization
    fig = create_visualization(results, problem_sizes, sparsities, configs)
    
    # Print summary
    print_summary_table(results, problem_sizes, sparsities, configs)
    
    print(f"\nBenchmark completed! Check 'matrix_multiplication_benchmark.png' for the visualization.")