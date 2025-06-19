import torch
import time


def torch_column_block_sparse_matmul(data, coordinates, B, block_size):
    """
    Torch baseline for column block sparse matrix multiplication.
    
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
    
    C = torch.zeros((M, N), dtype=data.dtype, device=data.device)
    
    # Process each column block
    for j in range(num_blocks):
        col_start = j * block_size
        col_end = col_start + block_size
        
        # Get the sparse data for this column
        data_col = data[:, col_start:col_end]  # (occupancy*block_size, block_size)
        
        # Process each occupied block in this column
        for occ_idx in range(occupancy):
            block_row = coordinates[j][occ_idx]
            row_start = block_row * block_size
            row_end = row_start + block_size
            
            # Extract the block data
            data_start = occ_idx * block_size
            data_end = data_start + block_size
            block_data = data_col[data_start:data_end, :]  # (block_size, block_size)
            
            # Multiply with corresponding B block
            B_block = B[row_start:row_end, :]  # (block_size, N)
            result_block = block_data @ B_block  # (block_size, N)
            
            # Add to result
            C[col_start:col_end, :] += result_block
    
    return C


def full_tensor(data, coordinates, block_size):
    """
    Convert column block sparse format to full dense tensor.
    
    Args:
        data: Compressed sparse data tensor (occupancy*block_size, num_blocks*block_size)
        coordinates: List of lists containing block row indices for each column
        block_size: Size of each block
    
    Returns:
        Dense tensor representation
    """
    rows, cols = data.shape
    num_blocks = cols // block_size
    occupancy = rows // block_size
    
    base_tensor = torch.zeros((num_blocks * block_size, num_blocks * block_size), 
                            dtype=data.dtype, device=data.device)
    
    for j in range(num_blocks):
        for i in range(occupancy):
            data_slice = data[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            row_start = block_size * coordinates[j][i]
            base_tensor[row_start:row_start+block_size, j*block_size:(j+1)*block_size] = data_slice
    
    return base_tensor


def bench_torch_column_sparse(data, coordinates, B, block_size, iters=20):
    """Benchmark the PyTorch column block sparse implementation."""
    if B.device.type == "cuda":
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            _ = torch_column_block_sparse_matmul(data, coordinates, B, block_size)
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) / iters
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = torch_column_block_sparse_matmul(data, coordinates, B, block_size)
        return (time.perf_counter() - t0) * 1e3 / iters


def bench_dense_baseline(A_full, B, iters=20):
    """Benchmark dense matrix multiplication baseline."""
    if B.device.type == "cuda":
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            _ = A_full @ B
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) / iters
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = A_full @ B
        return (time.perf_counter() - t0) * 1e3 / iters