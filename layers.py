import torch
from autograd import block_sparse_matmul_autograd
from block_mask import BlockMask
from block_sparse_matrix import BlockSparseMatrix
from einops import rearrange
import math


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
            repeat=self.__calculate_repeat(),
            device=device
        )
        
        # Create the sparse weight matrix (transposed for efficiency: n_in x n_out)
        self.weight_matrix = BlockSparseMatrix(block_size, block_size, self.block_mask, device, dtype)
        
        # Register the weight data as a parameter (create a separate parameter tensor)
        self.weight = torch.nn.Parameter(torch.empty_like(self.weight_matrix._data))
        
        # Initialize weights
        self._init_weights()

    def __calculate_repeat(self):
        n_blocks_in = self.n_in // self.block_size
        n_blocks_out = self.n_out // self.block_size
        gcd = math.gcd(n_blocks_in, n_blocks_out)
        return (n_blocks_in // gcd, n_blocks_out // gcd)

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


class BlockSparseMatmulFunctionEmulator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, block_mask):
        ctx.save_for_backward(A, B, block_mask)
        return torch.matmul(A, B)
    
    @staticmethod
    def backward(ctx, grad_output):
        A, B, block_mask = ctx.saved_tensors
        row_size = B.shape[0] // block_mask.shape[0]
        col_size = B.shape[1] // block_mask.shape[1]
        
        # Gradient w.r.t. A: grad_A = grad_output @ B.T
        grad_A = torch.matmul(grad_output, B.T)
        
        # Gradient w.r.t. B: grad_B = A.T @ grad_output
        grad_B = torch.matmul(A.T, grad_output)

        # grad_B needs masking
        grad_B = rearrange(grad_B, '(b r) (c s) -> b c r s', r=row_size, s=col_size)
        grad_B[~block_mask] = 0
        grad_B = rearrange(grad_B, 'b c r s -> (b r) (c s)')
        
        # Return gradients in same order as forward inputs
        return grad_A, grad_B, None


class SparseLinearEmulater(torch.nn.Module):
    def __init__(self, source: torch.nn.Module):
        super().__init__()
        self.matrix = torch.nn.Parameter(source.weight_matrix.get_full())
        self.block_mask = source.block_mask.mask

    def forward(self, x):
        return BlockSparseMatmulFunctionEmulator.apply(x, self.matrix, self.block_mask)
