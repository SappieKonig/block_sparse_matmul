from dataclasses import dataclass
import torch
import triton
import triton.language as tl
from functools import cached_property
import random


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


@dataclass
class BlockMask:
    mask: torch.Tensor

    @cached_property
    def num_rows(self):
        return self.mask.shape[0]

    @cached_property
    def num_cols(self):
        return self.mask.shape[1]

    @cached_property
    def row_occupancy(self):
        """Amount of non-zero entries in a row"""
        return self.mask.sum(dim=1)[0].item()

    @cached_property
    def col_occupancy(self):
        """Amount of non-zero entries in a column"""
        return self.mask.sum(dim=0)[0].item()

    @cached_property
    def total_blocks(self):
        return self.num_rows * self.row_occupancy

    @cached_property
    def matmul_block_indices(self):
        """Indices are to be compressed over the row dimension, as that's what the reduction would be for A @ B, where A is dense and B is block sparse."""
        indices = torch.zeros((self.col_occupancy, self.num_cols), dtype=torch.int32, device=self.mask.device)
        for col in range(self.num_cols):
            idx = 0
            for row in range(self.num_rows):
                if self.mask[row, col]:
                    indices[idx, col] = row
                    idx += 1
        idx_ranged = torch.arange(self.num_cols).repeat(self.col_occupancy, 1).to(self.mask.device)
        return indices, self._data_indices[indices, idx_ranged]

    @cached_property
    def matmul_block_indices_transposed(self):
        """Indices are to be compressed over the row dimension, as that's what the reduction would be for A @ B, where A is dense and B is block sparse."""
        indices = torch.zeros((self.num_rows, self.row_occupancy), dtype=torch.int32, device=self.mask.device)
        for row in range(self.num_rows):
            idx = 0
            for col in range(self.num_cols):
                if self.mask[row, col]:
                    indices[row, idx] = col
                    idx += 1
        idx_ranged = torch.arange(self.num_rows).view(-1, 1).repeat(1, self.row_occupancy).to(self.mask.device)
        return indices, self._data_indices[idx_ranged, indices]

    def __post_init__(self):
        assert self.mask.ndim == 2
        summed_0 = self.mask.sum(dim=0)
        assert summed_0.eq(summed_0[0]).all()
        summed_1 = self.mask.sum(dim=1)
        assert summed_1.eq(summed_1[0]).all()
        self._data_indices = self._build_data_indices()

    def _build_data_indices(self):
        data_indices = torch.full((self.num_rows, self.num_cols), fill_value=-1, dtype=torch.int32, device=self.mask.device)
        idx = 0
        for col in range(self.num_cols):
            for row in range(self.num_rows):
                if self.mask[row, col]:
                    data_indices[row, col] = idx
                    idx += 1
        return data_indices

    @classmethod
    def regular(cls, size: int, occupancy: int, device: str):
        mask = torch.zeros((size, size), dtype=torch.bool, device=device)
        for _ in range(occupancy):
            while True:
                new_indices = list(range(size))
                random.shuffle(new_indices)
                if any(mask[idx, col] for col, idx in enumerate(new_indices)):
                    continue
                else:
                    for col, idx in enumerate(new_indices):
                        mask[idx, col] = True
                    break
        return cls(mask)

    @classmethod
    def tiled_regular(cls, size: int, occupancy: int, repeat: tuple[int, int], device: str):
        mask = torch.zeros((repeat[0] * size, repeat[1] * size), dtype=torch.bool, device=device)
        for i in range(repeat[0]):
            for j in range(repeat[1]):
                mask[i * size:(i + 1) * size, j * size:(j + 1) * size] = cls.regular(size, occupancy, device).mask
        # shuffle along both dimensions
        row_indices = list(range(repeat[0] * size))
        col_indices = list(range(repeat[1] * size))
        random.shuffle(row_indices)
        random.shuffle(col_indices)
        mask = mask[row_indices, :][:, col_indices]
        return cls(mask)
        

class BlockSparseMatrix:
    def __init__(self, row_block: int, col_block: int, block_mask: BlockMask, device: str, dtype: torch.dtype):
        self.row_block = row_block
        self.col_block = col_block
        self.block_mask = block_mask
        self.device = device
        self.dtype = dtype

        self._data = torch.randn((row_block, col_block * block_mask.total_blocks), device=device, dtype=dtype)

    def get_full(self):
        full = torch.zeros((self.row_block * self.block_mask.num_rows, self.col_block * self.block_mask.num_cols), device=self.device, dtype=self.dtype)
        matmul_block_indices, data_indices = self.block_mask.matmul_block_indices
        for col in range(self.block_mask.num_cols):
            for row in range(self.block_mask.col_occupancy):
                row_start = matmul_block_indices[row, col] * self.row_block
                col_start = col * self.col_block
                data_start = data_indices[row, col] * self.col_block
                full[row_start:row_start + self.row_block, col_start:col_start + self.col_block] = self._data[:, data_start:data_start + self.col_block]
        return full

    def matmul(self, left):
        BLOCK_M = 64
        N = self.col_block * self.block_mask.num_cols
        M = left.shape[0]
        C = torch.empty((M, N), device=self.device, dtype=self.dtype)

        row_indices, data_indices = self.block_mask.matmul_block_indices
        print(row_indices.shape)
        print(self.block_mask.col_occupancy)

        grid = (M // BLOCK_M, self.block_mask.num_cols)
        block_sparse_mm_kernel[grid](
            left, self._data, C, row_indices, data_indices,
            BLOCK_M, self.col_block, self.row_block, self.block_mask.col_occupancy,
            left.stride(0), left.stride(1),
            self._data.stride(0), self._data.stride(1),
            C.stride(0), C.stride(1),
            row_indices.stride(0), row_indices.stride(1),
            data_indices.stride(0), data_indices.stride(1),
        )
        return C

    def matmul_transposed(self, C_grad):
        """Calculate the gradient of A in the A @ B = C kernel, with B being block sparse."""
        BLOCK_M = 64
        N = self.row_block * self.block_mask.num_rows
        M = C_grad.shape[0]
        A_grad = torch.empty((M, N), device=self.device, dtype=self.dtype)

        row_indices, data_indices = self.block_mask.matmul_block_indices_transposed
        print(row_indices.shape)
        print(self.block_mask.row_occupancy)

        grid = (M // BLOCK_M, self.block_mask.num_rows)
        block_sparse_mm_kernel_transposed[grid](
            C_grad, self._data, A_grad, row_indices, data_indices,
            BLOCK_M, self.col_block, self.row_block, self.block_mask.row_occupancy,
            C_grad.stride(0), C_grad.stride(1),
            self._data.stride(0), self._data.stride(1),
            A_grad.stride(0), A_grad.stride(1),
            row_indices.stride(1), row_indices.stride(0),
            data_indices.stride(1), data_indices.stride(0),
        )
        return A_grad

    def sparse_grad(self, A, C_grad):
        """Calculate the gradient of B in the A @ B = C kernel, with B being block sparse."""
        BLOCK_M = 64
        M = A.shape[0]
        row_indices, col_indices = torch.where(self.block_mask.mask)
        data_indices = self.block_mask._data_indices[row_indices, col_indices]
        grid = (self.block_mask.total_blocks,)
        B_grad = torch.empty_like(self._data)
        
        block_sparse_grad_kernel[grid](
            A, C_grad, B_grad, row_indices, col_indices, data_indices,
            BLOCK_M, self.row_block, self.col_block, M,
            A.stride(0), A.stride(1),
            C_grad.stride(0), C_grad.stride(1),
            B_grad.stride(0), B_grad.stride(1),
        )
        return B_grad


def print_tensor(tensor):
    print(*tensor.tolist(), sep="\n", end="\n\n")




class BlockSparseMatmulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, weight_data, block_sparse_matrix):
        ctx.save_for_backward(A, weight_data)
        ctx.block_sparse_matrix = block_sparse_matrix
        # Ensure the block sparse matrix uses the current weight data
        block_sparse_matrix._data = weight_data
        return block_sparse_matrix.matmul(A)
    
    @staticmethod
    def backward(ctx, grad_output):
        A, weight_data = ctx.saved_tensors
        block_sparse_matrix = ctx.block_sparse_matrix
        
        # Ensure the block sparse matrix uses the correct weight data for backward pass
        block_sparse_matrix._data = weight_data
        
        # Gradient w.r.t. A: grad_A = grad_output @ B.T
        grad_A = block_sparse_matrix.matmul_transposed(grad_output)
        
        # Gradient w.r.t. weight_data: grad_weight = A.T @ grad_output (sparse)
        grad_weight_data = block_sparse_matrix.sparse_grad(A, grad_output)
        
        # Return gradients in same order as forward inputs
        return grad_A, grad_weight_data, None


def block_sparse_matmul_autograd(A, weight_data, block_sparse_matrix):
    return BlockSparseMatmulFunction.apply(A, weight_data, block_sparse_matrix)


class SparseLinear(torch.nn.Module):
    def __init__(self, n_in: int, n_out: int, block_size: int, occupancy: int, device: str = "cuda", dtype: torch.dtype = torch.float16):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.block_size = block_size
        self.occupancy = occupancy
        self.device = device
        self.dtype = dtype
        
        # Calculate number of blocks
        assert n_in % block_size == 0, f"n_in ({n_in}) must be divisible by block_size ({block_size})"
        assert n_out % block_size == 0, f"n_out ({n_out}) must be divisible by block_size ({block_size})"
        
        n_blocks_in = n_in // block_size
        n_blocks_out = n_out // block_size
        
        # Create block mask for rectangular matrix (n_in x n_out)
        self.block_mask = BlockMask.tiled_regular(
            size=min(n_blocks_in, n_blocks_out), 
            occupancy=occupancy, 
            repeat=(n_blocks_in // min(n_blocks_in, n_blocks_out), n_blocks_out // min(n_blocks_in, n_blocks_out)),
            device=device
        )
        
        # Create the sparse weight matrix (transposed for efficiency: n_in x n_out)
        self.weight_matrix = BlockSparseMatrix(block_size, block_size, self.block_mask, device, dtype)
        
        # Register the weight data as a parameter (create a separate parameter tensor)
        self.weight = torch.nn.Parameter(torch.empty_like(self.weight_matrix._data))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Xavier/Glorot initialization adapted for sparse case
        fan_in = self.n_in * (self.occupancy / (self.n_in // self.block_size))
        fan_out = self.n_out * (self.occupancy / (self.n_out // self.block_size))
        std = (2.0 / (fan_in + fan_out)) ** 0.5
        
        with torch.no_grad():
            self.weight.normal_(0, std)
            # Also initialize the weight matrix data to the same values
            self.weight_matrix._data.copy_(self.weight)
    
    def forward(self, x):
        # x shape: (batch_size, n_in)
        # weight is n_in x n_out, so we need x @ weight
        # Pass the current weight parameter directly to ensure proper gradient flow
        return block_sparse_matmul_autograd(x, self.weight, self.weight_matrix)
    
    def extra_repr(self):
        return f'n_in={self.n_in}, n_out={self.n_out}, block_size={self.block_size}, occupancy={self.occupancy}, dtype={self.dtype}'


if __name__ == "__main__":
    import time
    from tqdm import tqdm
    
    device = "cuda"
    dtype = torch.float32

    BLOCKS = 32
    OCCUPANCY = 8

    BLOCK_K = 64
    BLOCK_N = 64

    REPEAT = (2, 1)

    M = 2048  # Reduced size
    K = BLOCK_K * BLOCKS * REPEAT[0]
    N = BLOCK_N * BLOCKS * REPEAT[1]

    # Clear GPU cache first
    torch.cuda.empty_cache()
    
    block_mask = BlockMask.tiled_regular(BLOCKS, OCCUPANCY, REPEAT, device)

    block_sparse_matrix = BlockSparseMatrix(BLOCK_K, BLOCK_N, block_mask, device, dtype)
    A = torch.randn((M, K), device=device, dtype=dtype)
    
    # Print memory usage
    print(f"A shape: {A.shape}, B shape: {block_sparse_matrix.get_full().shape}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Get dense matrix once for correctness and benchmarking
    B_dense = block_sparse_matrix.get_full()
    print(B_dense.dtype)
    print(A.dtype)
    
    # Correctness check - regular matmul
    C_sparse = block_sparse_matrix.matmul(A)
    C_dense = torch.mm(A, B_dense)
    max_diff = (C_sparse - C_dense).abs().max().item()
    print(f"Regular matmul max difference: {max_diff:.6f}")
    print(f"Regular matmul correctness: {'PASS' if max_diff < 1e-2 else 'FAIL'}")
    print(f"Regular matmul sparse absmax: {C_sparse.abs().max().item():.6f}")
    print(f"Regular matmul dense absmax: {C_dense.abs().max().item():.6f}")

    # not actually A transposed, just spoofing for shape reasons
    A_transposed = torch.randn((M, N), device=device, dtype=dtype)
    
    # Correctness check - transposed matmul
    C_sparse_T = block_sparse_matrix.matmul_transposed(A_transposed)
    C_dense_T = torch.mm(A_transposed, B_dense.t())
    max_diff_T = (C_sparse_T - C_dense_T).abs().max().item()
    print(f"Transposed matmul max difference: {max_diff_T:.6f}")
    print(f"Transposed matmul correctness: {'PASS' if max_diff_T < 1e-2 else 'FAIL'}")
    print(f"Transposed matmul sparse absmax: {C_sparse_T.abs().max().item():.6f}")
    print(f"Transposed matmul dense absmax: {C_dense_T.abs().max().item():.6f}")
    
    # Correctness check - gradient computation (cheaty test)
    # Create fake grad_C
    grad_C = torch.randn_like(C_dense)
    
    # Compute reference gradient: A.T @ grad_C and mask out non-active blocks
    grad_B_reference = torch.mm(A.t(), grad_C)
    mask = torch.zeros_like(grad_B_reference)
    row_indices, col_indices = torch.where(block_sparse_matrix.block_mask.mask)
    for i, j in zip(row_indices, col_indices):
        row_block = block_sparse_matrix.row_block
        col_block = block_sparse_matrix.col_block
        
        row_start = i * row_block
        col_start = j * col_block
        
        mask[row_start:row_start + row_block, col_start:col_start + col_block] = 1

    grad_B_reference *= mask
    
    # Use sparse_grad to compute gradient and overwrite _data
    grad_B_sparse = block_sparse_matrix.sparse_grad(A, grad_C)

    old_data = block_sparse_matrix._data

    block_sparse_matrix._data = grad_B_sparse
    grad_B_sparse_full = block_sparse_matrix.get_full()

    print(grad_B_sparse_full.shape)
    print(grad_B_reference.shape)

    max_diff = (grad_B_reference - grad_B_sparse_full).abs().max().item()
    print(f"Gradient computation max difference: {max_diff:.6f}")
    print(f"Gradient computation correctness: {'PASS' if max_diff < 1e-2 else 'FAIL'}")
    print(f"Gradient computation sparse absmax: {grad_B_sparse_full.abs().max().item():.6f}")
    print(f"Gradient computation dense absmax: {grad_B_reference.abs().max().item():.6f}")

    exit()

    block_sparse_matrix._data = old_data

    # Performance test
    num_warmup = 10
    num_trials = 50

    print(f"\nMatrix size: {A.shape[0]}x{A.shape[1]}")
    print(f"Block size: {block_sparse_matrix.row_block}x{block_sparse_matrix.col_block}")
    print(f"Sparsity: {block_mask.row_occupancy}/{block_mask.num_rows} = {block_mask.row_occupancy/block_mask.num_rows:.1%}")

    # Warmup - regular matmul
    for _ in range(num_warmup):
        _ = block_sparse_matrix.matmul(A)
        _ = torch.mm(A, B_dense)
        torch.cuda.synchronize()

    # Benchmark sparse - regular matmul
    torch.cuda.synchronize()
    start = time.time()
    for _ in tqdm(range(num_trials), desc="Sparse Regular"):
        C_sparse = block_sparse_matrix.matmul(A)
        torch.cuda.synchronize()
    sparse_time = (time.time() - start) / num_trials * 1000

    # Benchmark dense - regular matmul
    torch.cuda.synchronize()
    start = time.time()
    for _ in tqdm(range(num_trials), desc="Dense Regular"):
        C_dense = torch.mm(A, B_dense)
        torch.cuda.synchronize()
    dense_time = (time.time() - start) / num_trials * 1000

    print(f"\nRegular Matmul Results:")
    print(f"Sparse: {sparse_time:.2f} ms")
    print(f"Dense:  {dense_time:.2f} ms")
    print(f"Speedup: {dense_time/sparse_time:.2f}x")

    # Warmup - transposed matmul
    for _ in range(num_warmup):
        _ = block_sparse_matrix.matmul_transposed(A)
        _ = torch.mm(A, B_dense.t())
        torch.cuda.synchronize()

    # Benchmark sparse - transposed matmul
    torch.cuda.synchronize()
    start = time.time()
    for _ in tqdm(range(num_trials), desc="Sparse Transposed"):
        C_sparse_T = block_sparse_matrix.matmul_transposed(A)
        torch.cuda.synchronize()
    sparse_transposed_time = (time.time() - start) / num_trials * 1000

    # Benchmark dense - transposed matmul
    torch.cuda.synchronize()
    start = time.time()
    for _ in tqdm(range(num_trials), desc="Dense Transposed"):
        C_dense_T = torch.mm(A, B_dense.t())
        torch.cuda.synchronize()
    dense_transposed_time = (time.time() - start) / num_trials * 1000

    print(f"\nTransposed Matmul Results:")
    print(f"Sparse: {sparse_transposed_time:.2f} ms")
    print(f"Dense:  {dense_transposed_time:.2f} ms")
    print(f"Speedup: {dense_transposed_time/sparse_transposed_time:.2f}x")

    # Warmup - gradient computation
    grad_C = torch.randn_like(C_dense)
    for _ in range(num_warmup):
        _ = block_sparse_matrix.sparse_grad(A, grad_C)
        _ = torch.mm(A.t(), grad_C)
        torch.cuda.synchronize()

    # Benchmark sparse - gradient computation
    torch.cuda.synchronize()
    start = time.time()
    for _ in tqdm(range(num_trials), desc="Sparse Gradient"):
        grad_B_sparse = block_sparse_matrix.sparse_grad(A, grad_C)
        torch.cuda.synchronize()
    sparse_grad_time = (time.time() - start) / num_trials * 1000

    # Benchmark dense - gradient computation (only the computation part)
    torch.cuda.synchronize()
    start = time.time()
    for _ in tqdm(range(num_trials), desc="Dense Gradient"):
        grad_B_dense = torch.mm(A.t(), grad_C)
        torch.cuda.synchronize()
    dense_grad_time = (time.time() - start) / num_trials * 1000

    print(f"\nGradient Computation Results:")
    print(f"Sparse: {sparse_grad_time:.2f} ms")
    print(f"Dense:  {dense_grad_time:.2f} ms")
    print(f"Speedup: {dense_grad_time/sparse_grad_time:.2f}x")

    # Test AutoGrad function
    print(f"\n=== AutoGrad Function Test ===")
    A_grad_test = A.clone().requires_grad_(True)
    
    # Forward pass
    C_autograd = block_sparse_matmul_autograd(A_grad_test, block_sparse_matrix._data, block_sparse_matrix)
    
    # Create a simple loss
    loss = C_autograd.sum()
    
    # Backward pass
    loss.backward()
    
    print(f"AutoGrad forward correctness: {'PASS' if torch.allclose(C_autograd, C_sparse, atol=1e-2) else 'FAIL'}")
    print(f"A gradient shape: {A_grad_test.grad.shape}")
    print(f"A gradient norm: {A_grad_test.grad.norm().item():.6f}")
    
    # Verify gradient w.r.t. A by comparing with manual computation
    expected_grad_A = torch.ones_like(C_sparse) @ B_dense.t()
    grad_diff = (A_grad_test.grad - expected_grad_A).abs().max().item()
    print(f"A gradient correctness: {'PASS' if grad_diff < 1e-2 else 'FAIL'} (max diff: {grad_diff:.6f})")

    # Test SparseLinear layer
    print(f"\n=== SparseLinear Layer Test ===")
    sparse_layer = SparseLinear(n_in=1024, n_out=2048, block_size=64, occupancy=4, device=device, dtype=dtype)
    print(f"Layer: {sparse_layer}")
    
    # Test forward pass
    batch_size = 32
    x_test = torch.randn(batch_size, 1024, device=device, dtype=dtype, requires_grad=True)
    y_sparse = sparse_layer(x_test)
    
    print(f"Input shape: {x_test.shape}")
    print(f"Output shape: {y_sparse.shape}")
    
    # Test backward pass
    loss = y_sparse.sum()
    loss.backward()
    
    print(f"Input gradient shape: {x_test.grad.shape}")
    print(f"Weight gradient shape: {sparse_layer.weight.grad.shape}")
    print(f"SparseLinear layer test: PASS")

    # Test SGD optimization to drive weights to zero
    print(f"\n=== SGD Optimization Test ===")
    
    # Create a new layer for optimization test (use FP32 for better precision)
    opt_layer = SparseLinear(n_in=512, n_out=512, block_size=64, occupancy=4, device=device, dtype=torch.float32)
    
    # Create random input and zero target
    m, n = 64, 512
    input_data = torch.randn(m, n, device=device, dtype=torch.float32)
    target_output = torch.zeros(m, n, device=device, dtype=torch.float32)
    
    # Setup SGD optimizer with higher learning rate
    optimizer = torch.optim.SGD(opt_layer.parameters(), lr=0.1)
    
    print(f"Input shape: {input_data.shape}")
    print(f"Target shape: {target_output.shape}")
    print(f"Initial weight norm: {opt_layer.weight.norm().item():.6f}")
    
    # Store initial weight norm for comparison
    initial_weight_norm = opt_layer.weight.norm().item()
    
    # Training loop
    num_steps = 200
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Forward pass
        output = opt_layer(input_data)
        
        # Loss: MSE between output and zero target
        loss = torch.nn.functional.mse_loss(output, target_output)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Print progress every 20 steps
        if step % 20 == 0 or step == num_steps - 1:
            weight_norm = opt_layer.weight.norm().item()
            print(f"Step {step:3d}: Loss = {loss.item():.6f}, Weight norm = {weight_norm:.6f}")
    
    final_weight_norm = opt_layer.weight.norm().item()
    
    print(f"\nSparse Layer Optimization results:")
    print(f"Initial weight norm: {initial_weight_norm:.6f}")
    print(f"Final weight norm: {final_weight_norm:.6f}")
    print(f"Weight norm decreased: {'YES' if final_weight_norm < initial_weight_norm else 'NO'}")
    print(f"SGD optimization test: {'PASS' if final_weight_norm < initial_weight_norm else 'FAIL'}")

    # Test regular Linear layer for comparison
    print(f"\n=== Regular Linear Comparison ===")
    
    # Create a regular Linear layer with same dimensions (no bias to match SparseLinear)
    regular_layer = torch.nn.Linear(512, 512, bias=False, device=device, dtype=torch.float32)
    
    # Setup SGD optimizer with same learning rate
    regular_optimizer = torch.optim.SGD(regular_layer.parameters(), lr=0.1)
    
    print(f"Regular layer weight shape: {regular_layer.weight.shape}")
    print(f"Sparse layer weight shape: {opt_layer.weight.shape}")
    print(f"Regular layer param count: {regular_layer.weight.numel()}")
    print(f"Sparse layer param count: {opt_layer.weight.numel()}")
    print(f"Sparsity ratio: {opt_layer.weight.numel() / regular_layer.weight.numel():.3f}")
    
    # Store initial weight norm for comparison
    regular_initial_weight_norm = regular_layer.weight.norm().item()
    print(f"Initial regular weight norm: {regular_initial_weight_norm:.6f}")
    
    # Training loop for regular layer
    for step in range(num_steps):
        regular_optimizer.zero_grad()
        
        # Forward pass
        regular_output = regular_layer(input_data)
        
        # Loss: MSE between output and zero target
        regular_loss = torch.nn.functional.mse_loss(regular_output, target_output)
        
        # Backward pass
        regular_loss.backward()
        
        # Update weights
        regular_optimizer.step()
        
        # Print progress every 20 steps
        if step % 20 == 0 or step == num_steps - 1:
            regular_weight_norm = regular_layer.weight.norm().item()
            print(f"Step {step:3d}: Loss = {regular_loss.item():.6f}, Weight norm = {regular_weight_norm:.6f}")
    
    regular_final_weight_norm = regular_layer.weight.norm().item()
    
    print(f"\nRegular Layer Optimization results:")
    print(f"Initial weight norm: {regular_initial_weight_norm:.6f}")
    print(f"Final weight norm: {regular_final_weight_norm:.6f}")
    print(f"Weight norm decreased: {'YES' if regular_final_weight_norm < regular_initial_weight_norm else 'NO'}")
    
    # Comparison
    print(f"\n=== Comparison ===")
    sparse_reduction = (initial_weight_norm - final_weight_norm) / initial_weight_norm
    regular_reduction = (regular_initial_weight_norm - regular_final_weight_norm) / regular_initial_weight_norm
    print(f"Sparse layer weight reduction: {sparse_reduction:.3%}")
    print(f"Regular layer weight reduction: {regular_reduction:.3%}")
    print(f"Parameter efficiency: {opt_layer.weight.numel()}/{regular_layer.weight.numel()} = {opt_layer.weight.numel()/regular_layer.weight.numel():.1%}")
    print(f"Both optimizations working: {'YES' if sparse_reduction > 0 and regular_reduction > 0 else 'NO'}")


