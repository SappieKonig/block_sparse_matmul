import triton
import triton.language as tl


@triton.jit
def block_sparse_mm_kernel(
        A_ptr, B_data_ptr, C_ptr, row_indices_ptr, data_indices_ptr,
        block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr, occupancy: tl.constexpr,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_ri_k, stride_ri_n,
        stride_di_k, stride_di_n):

    pid_m = tl.program_id(0)  # batch block
    pid_n = tl.program_id(1)  # output column block
    
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    
    for sub_k in range(occupancy):
        block_idx = pid_n * stride_ri_n + sub_k * stride_ri_k
        row_block_idx = tl.load(row_indices_ptr + block_idx)
        
        # Load A tile
        offs_k = row_block_idx * block_k + tl.arange(0, block_k)
        A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_tile = tl.load(A_ptrs)
        
        # Load B tile
        block_idx = pid_n * stride_di_n + sub_k * stride_di_k
        data_col_start = tl.load(data_indices_ptr + block_idx) * block_n
        offs_b_k = tl.arange(0, block_k)
        offs_b_n = data_col_start + tl.arange(0, block_n)
        B_ptrs = B_data_ptr + offs_b_k[:, None] * stride_bk + offs_b_n[None, :] * stride_bn
        b_tile = tl.load(B_ptrs)
        
        acc += tl.dot(a_tile, b_tile)
    
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc.to(tl.float16))


@triton.jit
def block_sparse_mm_kernel_transposed(
        A_ptr, B_data_ptr, C_ptr, row_indices_ptr, data_indices_ptr,
        block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr, occupancy: tl.constexpr,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_ri_k, stride_ri_n,
        stride_di_k, stride_di_n):

    pid_m = tl.program_id(0)  # batch block
    pid_n = tl.program_id(1)  # output column block
    
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    
    for sub_k in range(occupancy):
        block_idx = pid_n * stride_ri_n + sub_k * stride_ri_k
        row_block_idx = tl.load(row_indices_ptr + block_idx)
        
        # Load A tile
        offs_k = row_block_idx * block_k + tl.arange(0, block_k)
        A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_tile = tl.load(A_ptrs)
        
        # Load B tile
        block_idx = pid_n * stride_di_n + sub_k * stride_di_k
        data_col_start = tl.load(data_indices_ptr + block_idx) * block_n
        offs_b_k = tl.arange(0, block_k)
        offs_b_n = data_col_start + tl.arange(0, block_n)
        B_ptrs = B_data_ptr + offs_b_k[:, None] * stride_bk + offs_b_n[None, :] * stride_bn
        b_tile = tl.load(B_ptrs)
        b_tile = tl.trans(b_tile)
        
        acc += tl.dot(a_tile, b_tile)
    
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc.to(tl.float16))


@triton.jit
def block_sparse_grad_kernel(
    A_ptr, C_grad_ptr, B_grad_ptr, row_indices_ptr, col_indices_ptr, data_indices_ptr,
    block_m: tl.constexpr, block_k: tl.constexpr, block_n: tl.constexpr, m: tl.constexpr,
    stride_a_m, stride_a_k,
    stride_c_grad_m, stride_c_grad_n,
    stride_b_grad_k, stride_b_grad_n,
):
    """A sparse gradient kernel to calculate the gradient of B in the A @ B = C kernel, with B being block sparse.
    
    Computes gradient w.r.t. B: grad_B = left.T @ right
    Each program handles one sparse block location.
    """
    
    # Each program handles one sparse block
    block_id = tl.program_id(0)
    
    # Get the row and column indices for this block (1D tensors from torch.where)
    row_block = tl.load(row_indices_ptr + block_id)
    col_block = tl.load(col_indices_ptr + block_id)
    data_block = tl.load(data_indices_ptr + block_id)
    
    # Compute offsets for this block in the full matrices
    offs_k = row_block * block_k + tl.arange(0, block_k)
    offs_n = col_block * block_n + tl.arange(0, block_n)
    
    # Initialize accumulator for this block: left.T @ right
    acc = tl.zeros((block_k, block_n), dtype=tl.float32)
    
    for m_start in range(0, m, block_m):
        offs_m = m_start + tl.arange(0, block_m)
        
        # Load left tile: left[offs_m, offs_k] (A matrix)
        left_ptrs = A_ptr + offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
        left_tile = tl.load(left_ptrs)
        
        # Load right tile: right[offs_m, offs_n] (grad_C matrix)
        right_ptrs = C_grad_ptr + offs_m[:, None] * stride_c_grad_m + offs_n[None, :] * stride_c_grad_n
        right_tile = tl.load(right_ptrs)
        
        # Accumulate: left.T @ right
        acc += tl.dot(tl.trans(left_tile), right_tile)
    
    # Store the result in the output gradient matrix
    offs_n = data_block * block_n + tl.arange(0, block_n)
    B_grad_ptrs = B_grad_ptr + tl.arange(0, block_k)[:, None] * stride_b_grad_k + offs_n[None, :] * stride_b_grad_n
    tl.store(B_grad_ptrs, acc.to(tl.float16))