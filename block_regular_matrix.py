import torch
import random
from dataclasses import dataclass
from torch.autograd import Function


@dataclass
class Coordinate:
    row: int
    col: int


@dataclass
class Block:
    coord: Coordinate
    index: int


class BlockRegularMatrix:
    def __init__(self, block_size: int, num_blocks: int, k: int, device: str, dtype: torch.dtype):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.k = k
        self.device = device
        self.dtype = dtype
        self.data = torch.randn((block_size, block_size * k * num_blocks), dtype=dtype, device=device)
        self.coordinates = self.create_blueprint()

    def __repr__(self):
        FULL = "██"
        EMPTY = "  "

        example = torch.zeros((self.num_blocks, self.num_blocks))
        for block in self.coordinates:
            example[block.coord.row, block.coord.col] = 1
        example = example.tolist()
        example_string = "\n".join(["".join([FULL if x else EMPTY for x in y]) for y in example])
        return example_string

    def create_blueprint(self):
        blueprint = []
        for i in range(self.k):
            indices = list(range(self.num_blocks))
            while True:
                random.shuffle(indices)
                for used in blueprint:
                    if any(x == y for x, y in zip(used, indices)):
                        break
                else:
                    blueprint.append(indices)
                    break
        coordinates = []
        for l in blueprint:
            for col, row in enumerate(l):
                coordinates.append(Coordinate(row, col))
        # sort by column
        coordinates = sorted(coordinates, key=lambda x: (x.col, x.row))
        blocks = [Block(coord, i) for i, coord in enumerate(coordinates)]
        return blocks

    def as_full(self):
        full = torch.zeros((self.num_blocks * self.block_size, self.num_blocks * self.block_size), dtype=self.dtype, device=self.device)
        for block in self.coordinates:
            idx = block.index
            row, col = block.coord.row, block.coord.col
            row_start = row * self.block_size
            col_start = col * self.block_size
            full[row_start:row_start + self.block_size, col_start:col_start + self.block_size] = self.data[:, idx * self.block_size:(idx + 1) * self.block_size]
        return full


import triton
import triton.language as tl


@triton.jit
def block_regular_mm_kernel(
        A_ptr,              # float16 [batch_size, num_blocks*block_size]
        B_data_ptr,         # float16 [block_size, block_size * k * num_blocks] (sparse data)
        B_indices_ptr,      # int32 [k * num_blocks] (block indices)
        C_ptr,              # float16 [batch_size, num_blocks*block_size]
        batch_size, num_blocks, k, block_size,
        stride_am, stride_ak,      # A strides
        stride_bm, stride_bn,      # B_data strides  
        stride_cm, stride_cn,      # C strides
        BLOCK: tl.constexpr):

    # Identify which block we're working on
    pid_m = tl.program_id(0)  # batch block
    pid_n = tl.program_id(1)  # output column block
    
    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
    
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    
    # For each block in this column, find which blocks are non-zero
    for i in range(k):
        # Get the block index for this position in the column
        block_idx = pid_n * k + i
        row_block_idx = tl.load(B_indices_ptr + block_idx)
        
        # Load A tile from the corresponding row block
        offs_k = row_block_idx * BLOCK + tl.arange(0, BLOCK)
        A_ptrs = (A_ptr + 
                  offs_m[:, None] * stride_am + 
                  offs_k[None, :] * stride_ak)
        a_tile = tl.load(A_ptrs)
        
        # Load B tile from the sparse data
        data_col_start = block_idx * BLOCK
        offs_b_m = tl.arange(0, BLOCK)
        offs_b_n = data_col_start + tl.arange(0, BLOCK)
        B_ptrs = (B_data_ptr + 
                  offs_b_m[:, None] * stride_bm + 
                  offs_b_n[None, :] * stride_bn)
        b_tile = tl.load(B_ptrs)
        
        # Accumulate A @ B_sparse
        acc += tl.dot(a_tile, b_tile)
    
    # Write result
    C_ptrs = (C_ptr + 
              offs_m[:, None] * stride_cm + 
              offs_n[None, :] * stride_cn)
    tl.store(C_ptrs, acc.to(tl.float16))


def block_regular_mm(A: torch.Tensor, B_sparse: 'BlockRegularMatrix') -> torch.Tensor:
    batch_size, A_cols = A.shape
    assert A_cols == B_sparse.num_blocks * B_sparse.block_size
    
    # Create output tensor
    C = torch.zeros((batch_size, B_sparse.num_blocks * B_sparse.block_size), 
                    dtype=A.dtype, device=A.device)
    
    # Create indices tensor for the kernel
    indices = torch.tensor([block.coord.row for block in B_sparse.coordinates], 
                          dtype=torch.int32, device=A.device)
    
    # Launch kernel
    grid = (batch_size // B_sparse.block_size, B_sparse.num_blocks)
    
    block_regular_mm_kernel[grid](
        A, B_sparse.data, indices, C,
        batch_size, B_sparse.num_blocks, B_sparse.k, B_sparse.block_size,
        A.stride(0), A.stride(1),
        B_sparse.data.stride(0), B_sparse.data.stride(1),
        C.stride(0), C.stride(1),
        BLOCK=B_sparse.block_size
    )
    
    return C


@triton.jit
def block_regular_mm_backward_kernel(
        grad_C_ptr,         # float16 [batch_size, num_blocks*block_size] - incoming gradient
        A_ptr,              # float16 [batch_size, num_blocks*block_size] - original A input
        B_data_ptr,         # float16 [block_size, block_size * k * num_blocks] - sparse B data
        B_indices_ptr,      # int32 [k * num_blocks] - block indices
        grad_A_ptr,         # float16 [batch_size, num_blocks*block_size] - gradient wrt A
        grad_B_data_ptr,    # float16 [block_size, block_size * k * num_blocks] - gradient wrt B data
        batch_size, num_blocks, k, block_size,
        stride_gc_m, stride_gc_n,    # grad_C strides
        stride_am, stride_an,        # A strides
        stride_bm, stride_bn,        # B_data strides
        stride_ga_m, stride_ga_n,    # grad_A strides
        stride_gb_m, stride_gb_n,    # grad_B_data strides
        BLOCK: tl.constexpr):

    pid_m = tl.program_id(0)  # batch block
    pid_n = tl.program_id(1)  # column block
    
    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
    
    # Load grad_C tile
    grad_C_ptrs = (grad_C_ptr + 
                   offs_m[:, None] * stride_gc_m + 
                   offs_n[None, :] * stride_gc_n)
    grad_c_tile = tl.load(grad_C_ptrs)
    
    grad_a_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    
    # Compute gradients for each block in this column
    for i in range(k):
        block_idx = pid_n * k + i
        row_block_idx = tl.load(B_indices_ptr + block_idx)
        
        # Load B tile for grad_A computation
        data_col_start = block_idx * BLOCK
        offs_b_m = tl.arange(0, BLOCK)
        offs_b_n = data_col_start + tl.arange(0, BLOCK)
        B_ptrs = (B_data_ptr + 
                  offs_b_m[:, None] * stride_bm + 
                  offs_b_n[None, :] * stride_bn)
        b_tile = tl.load(B_ptrs)
        
        # grad_A += grad_C @ B^T
        grad_a_acc += tl.dot(grad_c_tile, tl.trans(b_tile))
        
        # For grad_B: grad_B += A^T @ grad_C
        # Load A tile
        offs_k = row_block_idx * BLOCK + tl.arange(0, BLOCK)
        A_ptrs = (A_ptr + 
                  offs_m[:, None] * stride_am + 
                  offs_k[None, :] * stride_an)
        a_tile = tl.load(A_ptrs)
        
        # Compute grad_B for this block
        grad_b_tile = tl.dot(tl.trans(a_tile), grad_c_tile)
        
        # Store grad_B
        grad_B_ptrs = (grad_B_data_ptr + 
                       offs_b_m[:, None] * stride_gb_m + 
                       offs_b_n[None, :] * stride_gb_n)
        tl.store(grad_B_ptrs, grad_b_tile.to(tl.float16))
    
    # Store grad_A - need to accumulate contributions from all column blocks
    # For now, store directly to the column block position
    offs_k = pid_n * BLOCK + tl.arange(0, BLOCK)
    grad_A_ptrs = (grad_A_ptr + 
                   offs_m[:, None] * stride_ga_m + 
                   offs_k[None, :] * stride_ga_n)
    tl.store(grad_A_ptrs, grad_a_acc.to(tl.float16))


class BlockRegularMatmul(Function):
    @staticmethod
    def forward(ctx, A, B_sparse):
        # Save for backward
        ctx.save_for_backward(A)
        ctx.B_sparse = B_sparse
        
        # Forward pass
        C = block_regular_mm(A, B_sparse)
        return C
    
    @staticmethod
    def backward(ctx, grad_C):
        A, = ctx.saved_tensors
        B_sparse = ctx.B_sparse
        
        # For now, use PyTorch's native operations for backward pass
        # This ensures correctness while we can optimize later
        
        # Compute grad_A = grad_C @ B_sparse^T
        B_full = B_sparse.as_full()
        grad_A = torch.mm(grad_C, B_full.t())
        
        # For grad_B, we need to compute A^T @ grad_C
        # and then extract only the relevant sparse blocks
        grad_B_full = torch.mm(A.t(), grad_C)
        
        # Extract gradients for sparse blocks and assign to B_sparse.data
        if B_sparse.data.requires_grad:
            grad_B_data = torch.zeros_like(B_sparse.data)
            for block in B_sparse.coordinates:
                idx = block.index
                row, col = block.coord.row, block.coord.col
                row_start = row * B_sparse.block_size
                col_start = col * B_sparse.block_size
                block_grad = grad_B_full[row_start:row_start + B_sparse.block_size, 
                                       col_start:col_start + B_sparse.block_size]
                grad_B_data[:, idx * B_sparse.block_size:(idx + 1) * B_sparse.block_size] = block_grad
            
            # Assign gradient to B_sparse.data
            if B_sparse.data.grad is None:
                B_sparse.data.grad = grad_B_data
            else:
                B_sparse.data.grad += grad_B_data
        
        return grad_A, None  # None for B_sparse since it's not a tensor


def block_regular_matmul_autograd(A, B_sparse):
    """Autograd-enabled version of block regular matrix multiplication"""
    return BlockRegularMatmul.apply(A, B_sparse)


def test_correctness():
    device = "cuda"
    dtype = torch.float16
    block_size = 64  # Reduced from 128 to save shared memory
    num_blocks = 4
    k = 2
    batch_size = 128  # Reduced batch size too
    
    # Create sparse matrix
    B_sparse = BlockRegularMatrix(block_size, num_blocks, k, device, dtype)
    
    # Create dense input matrix A
    A = torch.randn((batch_size, num_blocks * block_size), dtype=dtype, device=device)
    
    # Get full B matrix for comparison
    B_full = B_sparse.as_full()
    
    # Compute sparse result
    C_sparse = block_regular_mm(A, B_sparse)
    
    # Compute dense result
    C_full = torch.mm(A, B_full)

    print(C_sparse.abs().max())
    print(C_full.abs().max())
    
    # Check correctness
    max_diff = torch.max(torch.abs(C_sparse - C_full)).item()
    mean_diff = torch.mean(torch.abs(C_sparse - C_full)).item()
    
    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")
    print(f"Correctness test: {'PASSED' if max_diff < 1e-2 else 'FAILED'}")
    
    return max_diff < 1e-2


def test_gradient_correctness():
    device = "cuda"
    dtype = torch.float32  # Use float32 for better gradient precision
    block_size = 64
    num_blocks = 4
    k = 2
    batch_size = 128
    
    # Create sparse matrix
    B_sparse = BlockRegularMatrix(block_size, num_blocks, k, device, dtype)
    B_sparse.data.requires_grad_(True)  # Enable gradients for sparse data
    
    # Create input matrix A with gradient tracking
    A = torch.randn((batch_size, num_blocks * block_size), dtype=dtype, device=device, requires_grad=True)
    
    # Test forward pass first
    C_sparse = block_regular_matmul_autograd(A, B_sparse)
    
    # For comparison, we need a fresh A without gradients
    A_full = A.clone().detach().requires_grad_(True)
    B_full = B_sparse.as_full().detach().requires_grad_(True)  # Detach to make it a leaf tensor
    C_full = torch.mm(A_full, B_full)
    
    # Check forward pass correctness
    forward_diff = torch.max(torch.abs(C_sparse - C_full)).item()
    print(f"Forward pass max difference: {forward_diff}")
    
    # Create random gradient for backward pass
    grad_output = torch.randn_like(C_sparse)
    
    # Backward pass for sparse version
    C_sparse.backward(grad_output)
    grad_A_sparse = A.grad.clone()
    
    # Backward pass for dense version  
    C_full.backward(grad_output)
    grad_A_full = A_full.grad.clone()
    grad_B_full = B_full.grad.clone()
    
    # Check gradient correctness for A
    grad_A_diff = torch.max(torch.abs(grad_A_sparse - grad_A_full)).item()
    grad_A_mean_diff = torch.mean(torch.abs(grad_A_sparse - grad_A_full)).item()
    grad_A_sparse_absmax = torch.max(torch.abs(grad_A_sparse)).item()
    grad_A_full_absmax = torch.max(torch.abs(grad_A_full)).item()
    
    print(f"Gradient A sparse absmax: {grad_A_sparse_absmax}")
    print(f"Gradient A full absmax: {grad_A_full_absmax}")
    print(f"Gradient A max difference: {grad_A_diff}")
    print(f"Gradient A mean difference: {grad_A_mean_diff}")
    
    # For B gradient, we need to extract from full B and compare with sparse data
    # The sparse B gradient should match the corresponding blocks in the full gradient
    grad_B_sparse_reconstructed = torch.zeros_like(B_full)
    for block in B_sparse.coordinates:
        idx = block.index
        row, col = block.coord.row, block.coord.col
        row_start = row * block_size
        col_start = col * block_size
        grad_B_sparse_reconstructed[row_start:row_start + block_size, col_start:col_start + block_size] = \
            B_sparse.data.grad[:, idx * block_size:(idx + 1) * block_size] if B_sparse.data.grad is not None else 0
    
    # Compare only the non-zero blocks
    grad_B_diff = 0
    grad_B_count = 0
    for block in B_sparse.coordinates:
        row, col = block.coord.row, block.coord.col
        row_start = row * block_size
        col_start = col * block_size
        full_block = grad_B_full[row_start:row_start + block_size, col_start:col_start + block_size]
        sparse_block = grad_B_sparse_reconstructed[row_start:row_start + block_size, col_start:col_start + block_size]
        block_diff = torch.max(torch.abs(full_block - sparse_block)).item()
        grad_B_diff = max(grad_B_diff, block_diff)
        grad_B_count += 1
    
    # Print gradient magnitudes for B
    grad_B_sparse_absmax = torch.max(torch.abs(B_sparse.data.grad)).item() if B_sparse.data.grad is not None else 0
    grad_B_full_absmax = torch.max(torch.abs(grad_B_full)).item()
    
    print(f"Gradient B sparse absmax: {grad_B_sparse_absmax}")
    print(f"Gradient B full absmax: {grad_B_full_absmax}")
    print(f"Gradient B max difference: {grad_B_diff}")
    
    # Overall test result - more lenient thresholds for float32
    gradient_test_passed = (forward_diff < 0.1 and 
                           grad_A_diff < 1e-4 and 
                           grad_B_diff < 1e-4)
    
    print(f"\nGradient correctness test: {'PASSED' if gradient_test_passed else 'FAILED'}")
    
    return gradient_test_passed


if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float16
    block_size = 64  # Using smaller block size that works with hardware limits
    num_blocks = 16
    k = 4
    block_regular_matrix = BlockRegularMatrix(block_size, num_blocks, k, device, dtype)
    print(block_regular_matrix)
    print("\nRunning correctness test...")
    test_correctness()
    print("\nRunning gradient correctness test...")
    test_gradient_correctness()
