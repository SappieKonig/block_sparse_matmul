import torch
from block_sparse_matrix import BlockSparseMatrix


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
    