import torch
import time
from tqdm import tqdm
from block_regular_matrix import BlockRegularMatrix, block_regular_matmul_autograd

def benchmark_forward_backward(matrix_size: int, batch_size: int = 64, num_warmup: int = 3, num_trials: int = 10):
    device = "cuda"
    dtype = torch.float16
    
    # Block sparse matrix parameters
    block_size = 64
    num_blocks = matrix_size // block_size
    k = max(2, num_blocks // 4)  # Adjust sparsity based on size
    
    print(f"\n=== Matrix Size: {matrix_size}x{matrix_size} ===")
    print(f"Block size: {block_size}, Num blocks: {num_blocks}, k: {k}")
    print(f"Sparsity: {k / num_blocks:.2%}")
    
    # Warmup - Sparse
    for _ in tqdm(range(num_warmup), desc="Sparse warmup", leave=False):
        B_sparse = BlockRegularMatrix(block_size, num_blocks, k, device, dtype)
        B_sparse.data.requires_grad_(True)
        A_copy = torch.randn((batch_size, matrix_size), dtype=dtype, device=device, requires_grad=True)
        C_sparse = block_regular_matmul_autograd(A_copy, B_sparse)
        loss = C_sparse.sum()
        loss.backward()
        torch.cuda.synchronize()
    
    # Actual timing - Sparse forward
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in tqdm(range(num_trials), desc="Sparse forward", leave=False):
        B_sparse = BlockRegularMatrix(block_size, num_blocks, k, device, dtype)
        A_copy = torch.randn((batch_size, matrix_size), dtype=dtype, device=device)
        C_sparse = block_regular_matmul_autograd(A_copy, B_sparse)
        torch.cuda.synchronize()
    sparse_forward_time = (time.time() - start_time) / num_trials * 1000  # ms
    
    # Sparse forward + backward
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in tqdm(range(num_trials), desc="Sparse forward+backward", leave=False):
        B_sparse = BlockRegularMatrix(block_size, num_blocks, k, device, dtype)
        B_sparse.data.requires_grad_(True)
        A_copy = torch.randn((batch_size, matrix_size), dtype=dtype, device=device, requires_grad=True)
        C_sparse = block_regular_matmul_autograd(A_copy, B_sparse)
        loss = C_sparse.sum()
        loss.backward()
        torch.cuda.synchronize()
    sparse_forward_backward_time = (time.time() - start_time) / num_trials * 1000  # ms
    
    # Get one instance for dense comparison
    B_sparse_ref = BlockRegularMatrix(block_size, num_blocks, k, device, dtype)
    B_dense_template = B_sparse_ref.as_full()
    
    # Warmup - Dense
    for _ in tqdm(range(num_warmup), desc="Dense warmup", leave=False):
        B_dense = B_dense_template.clone().requires_grad_(True)
        A_copy = torch.randn((batch_size, matrix_size), dtype=dtype, device=device, requires_grad=True)
        C_dense = torch.mm(A_copy, B_dense)
        loss = C_dense.sum()
        loss.backward()
        torch.cuda.synchronize()
    
    # Dense forward
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in tqdm(range(num_trials), desc="Dense forward", leave=False):
        B_dense = B_dense_template.clone()
        A_copy = torch.randn((batch_size, matrix_size), dtype=dtype, device=device)
        C_dense = torch.mm(A_copy, B_dense)
        torch.cuda.synchronize()
    dense_forward_time = (time.time() - start_time) / num_trials * 1000  # ms
    
    # Dense forward + backward
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in tqdm(range(num_trials), desc="Dense forward+backward", leave=False):
        B_dense = B_dense_template.clone().requires_grad_(True)
        A_copy = torch.randn((batch_size, matrix_size), dtype=dtype, device=device, requires_grad=True)
        C_dense = torch.mm(A_copy, B_dense)
        loss = C_dense.sum()
        loss.backward()
        torch.cuda.synchronize()
    dense_forward_backward_time = (time.time() - start_time) / num_trials * 1000  # ms
    
    # Calculate speedup
    forward_speedup = dense_forward_time / sparse_forward_time
    backward_speedup = dense_forward_backward_time / sparse_forward_backward_time
    
    print(f"Forward pass:")
    print(f"  Sparse: {sparse_forward_time:.2f} ms")
    print(f"  Dense:  {dense_forward_time:.2f} ms")
    print(f"  Speedup: {forward_speedup:.2f}x")
    
    print(f"Forward + Backward pass:")
    print(f"  Sparse: {sparse_forward_backward_time:.2f} ms")
    print(f"  Dense:  {dense_forward_backward_time:.2f} ms")
    print(f"  Speedup: {backward_speedup:.2f}x")
    
    return {
        'matrix_size': matrix_size,
        'sparse_forward': sparse_forward_time,
        'dense_forward': dense_forward_time,
        'sparse_forward_backward': sparse_forward_backward_time,
        'dense_forward_backward': dense_forward_backward_time,
        'forward_speedup': forward_speedup,
        'backward_speedup': backward_speedup,
        'sparsity': k / num_blocks
    }

def main():
    print("Block Regular Matrix Performance Benchmark")
    print("=" * 50)
    
    sizes = [256, 1024, 2048]
    results = []
    
    for size in sizes:
        try:
            result = benchmark_forward_backward(size, batch_size=32, num_warmup=2, num_trials=5)
            results.append(result)
        except Exception as e:
            print(f"Error benchmarking size {size}: {e}")
            continue
    
    # Summary table
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Size':<6} {'Sparsity':<10} {'Forward':<15} {'F+B':<15} {'Forward':<10} {'F+B':<10}")
    print(f"{'':^6} {'':^10} {'Speedup':<15} {'Speedup':<15} {'Sparse(ms)':<10} {'Sparse(ms)':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['matrix_size']:<6} "
              f"{result['sparsity']:.1%}{'':^4} "
              f"{result['forward_speedup']:.2f}x{'':^10} "
              f"{result['backward_speedup']:.2f}x{'':^10} "
              f"{result['sparse_forward']:.1f}{'':^5} "
              f"{result['sparse_forward_backward']:.1f}")

if __name__ == "__main__":
    main()