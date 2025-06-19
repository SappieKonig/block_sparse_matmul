import torch
import triton
import triton.language as tl
import time


def get_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def column_block_sparse_kernel(
    data_ptr, coordinates_ptr, b_ptr, c_ptr,
    block_size: tl.constexpr, num_blocks: tl.constexpr, occupancy: tl.constexpr,
    M, N, K,
    stride_data_row, stride_data_col,
    stride_b_row, stride_b_col,
    stride_c_row, stride_c_col,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    """
    Column block sparse matrix multiplication kernel.
    
    Args:
        data_ptr: Compressed sparse data (occupancy*block_size, num_blocks*block_size)
        coordinates_ptr: Block coordinates for each column
        b_ptr: Dense matrix B
        c_ptr: Output matrix C
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    m_mask = offs_m < M
    n_mask = offs_n < N
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over blocks in the K dimension
    for block_k in range(num_blocks):
        k_start = block_k * block_size
        k_end = k_start + block_size
        
        # Process each occupancy slot for this block column
        for occ_idx in range(occupancy):
            # Load coordinate for this block column and occupancy slot
            coord_idx = block_k * occupancy + occ_idx
            block_row = tl.load(coordinates_ptr + coord_idx)
            
            # Calculate actual row range in the full matrix
            actual_k_start = block_row * block_size
            actual_k_end = actual_k_start + block_size
            
            # Load data from compressed format
            for k in range(0, block_size, BLOCK_SIZE_K):
                k_offs = k + tl.arange(0, BLOCK_SIZE_K)
                k_mask = k_offs < block_size
                
                # Data indices in compressed format
                data_k_offs = occ_idx * block_size + k_offs
                data_n_offs = k_start + tl.arange(0, BLOCK_SIZE_N)
                
                # B indices in original matrix
                b_k_offs = actual_k_start + k_offs
                
                data_ptrs = data_ptr + (data_k_offs[:, None] * stride_data_row + 
                                      (k_start + offs_n[None, :]) * stride_data_col)
                b_ptrs = b_ptr + ((actual_k_start + k_offs)[:, None] * stride_b_row + 
                                offs_n[None, :] * stride_b_col)
                
                data_mask = k_mask[:, None] & n_mask[None, :] & (offs_n[None, :] < (k_start + block_size))
                b_mask = k_mask[:, None] & n_mask[None, :] & (b_k_offs[:, None] < K)
                
                a = tl.load(data_ptrs, mask=data_mask, other=0.0)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)
                
                accumulator += tl.dot(a, b)
    
    c_ptrs = c_ptr + stride_c_row * offs_m[:, None] + stride_c_col * offs_n[None, :]
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_column_block_sparse_matmul(data, coordinates, B, block_size):
    """
    Perform column block sparse matrix multiplication using Triton.
    
    Args:
        data: Compressed sparse data tensor (occupancy*block_size, num_blocks*block_size)
        coordinates: List of lists containing block row indices for each column
        B: Dense matrix B
        block_size: Size of each block
    
    Returns:
        Result of sparse_matrix @ B
    """
    occupancy_bs, num_blocks_bs = data.shape
    occupancy = occupancy_bs // block_size
    num_blocks = num_blocks_bs // block_size
    
    K, N = B.shape
    M = num_blocks * block_size
    
    # Flatten coordinates to a tensor
    coords_flat = []
    for col_coords in coordinates:
        coords_flat.extend(col_coords)
    coordinates_tensor = torch.tensor(coords_flat, dtype=torch.int32, device=data.device)
    
    C = torch.zeros((M, N), dtype=torch.float32, device=data.device)
    
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 32, 32, 32
    grid = ((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    
    column_block_sparse_kernel[grid](
        data, coordinates_tensor, B, C,
        block_size, num_blocks, occupancy,
        M, N, K,
        data.stride(0), data.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return C.to(data.dtype)


def bench_triton_column_sparse(data, coordinates, B, block_size, iters=20):
    """Benchmark the Triton column block sparse implementation."""
    if B.device.type == "cuda":
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            _ = triton_column_block_sparse_matmul(data, coordinates, B, block_size)
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) / iters
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = triton_column_block_sparse_matmul(data, coordinates, B, block_size)
        return (time.perf_counter() - t0) * 1e3 / iters