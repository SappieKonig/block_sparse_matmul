from random import shuffle
import torch
from einops import rearrange
from triton_kernel import triton_column_block_sparse_matmul, bench_triton_column_sparse
from torch_baseline import torch_column_block_sparse_matmul, bench_torch_column_sparse, bench_dense_baseline, full_tensor


def generate_column_block_sparse_matrix(block_size: int, num_blocks: int, occupancy: int, device: str, dtype: torch.dtype):
    coordinates = []
    for _ in range(num_blocks):
        indices = list(range(num_blocks))
        shuffle(indices)
        coordinates.append(sorted(indices[:occupancy]))

    data = torch.randn((occupancy * block_size, num_blocks * block_size), dtype=dtype, device=device)
    return data, coordinates


if __name__ == "__main__":
    device = "cuda"
    dtype = torch.bfloat16
    block_size = 128
    num_blocks = 16
    occupancy = 4
    batch_size = 512
    
    # Generate column block sparse matrix
    data, coordinates = generate_column_block_sparse_matrix(block_size, num_blocks, occupancy, device, dtype)
    A_full = full_tensor(data, coordinates, block_size)
    
    # Generate dense B matrix
    K, N = A_full.shape[0], batch_size
    B = torch.randn((K, N), dtype=dtype, device=device)
    
    print(f"Data shape: {data.shape}")
    print(f"A_full shape: {A_full.shape}")
    print(f"B shape: {B.shape}")
    print(f"Sparsity: {1 - (occupancy / num_blocks):.2%}")
    
    # Benchmark dense baseline
    dense_ms = bench_dense_baseline(A_full, B)
    
    # Benchmark PyTorch column sparse
    torch_ms = bench_torch_column_sparse(data, coordinates, B, block_size)
    torch_result = torch_column_block_sparse_matmul(data, coordinates, B, block_size)
    
    # Benchmark Triton column sparse  
    triton_ms = bench_triton_column_sparse(data, coordinates, B, block_size)
    triton_result = triton_column_block_sparse_matmul(data, coordinates, B, block_size)
    
    # Verify correctness
    dense_result = A_full @ B
    torch_diff = (dense_result.float() - torch_result.float()).abs().max().item()
    triton_diff = (dense_result.float() - triton_result.float()).abs().max().item()
    
    print(f"\nDevice: {device}")
    print(f"Dense baseline     : {dense_ms:.4f} ms")
    print(f"PyTorch sparse     : {torch_ms:.4f} ms   (speed-up ×{dense_ms/torch_ms:.2f})")
    print(f"Triton sparse      : {triton_ms:.4f} ms   (speed-up ×{dense_ms/triton_ms:.2f})")
    print(f"PyTorch max error  : {torch_diff:e}")
    print(f"Triton max error   : {triton_diff:e}")
    
    # Show sparsity pattern
    print(f"\nSparsity pattern visualization:")
    A_pattern = (A_full != 0).float()
    A_pattern = rearrange(A_pattern, "(a b) (c d) -> a c (b d)", b=block_size, d=block_size)
    A_pattern = torch.max(A_pattern, dim=-1).values
    print("Block occupancy pattern:")
    for row in A_pattern.tolist():
        print(''.join('█' if x > 0 else '·' for x in row))