from block_mask import BlockMask
import torch
from kernels import block_sparse_mm_kernel, block_sparse_mm_kernel_transposed, block_sparse_grad_kernel


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
        BLOCK_M = 32
        N = self.col_block * self.block_mask.num_cols
        M = left.shape[0]
        C = torch.empty((M, N), device=self.device, dtype=self.dtype)

        row_indices, data_indices = self.block_mask.matmul_block_indices

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
        BLOCK_M = 32
        N = self.row_block * self.block_mask.num_rows
        M = C_grad.shape[0]
        A_grad = torch.empty((M, N), device=self.device, dtype=self.dtype)

        row_indices, data_indices = self.block_mask.matmul_block_indices_transposed

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
        BLOCK_M = 32
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