import pytest
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from layers import SparseLinear, SparseLinearEmulater


class TestSparseLinearCorrectness:
    """Test correctness of SparseLinear against dense emulator."""
    
    # Test parameters - easy to extend by adding values to these lists
    N_IN_VALUES = [512, 1024]
    N_OUT_VALUES = [512, 1024] 
    BLOCK_SIZE_VALUES = [64]
    OCCUPANCY_VALUES = [2, 4]
    BATCH_SIZE_VALUES = [128]
    
    @pytest.mark.parametrize("n_in", N_IN_VALUES)
    @pytest.mark.parametrize("n_out", N_OUT_VALUES)
    @pytest.mark.parametrize("block_size", BLOCK_SIZE_VALUES)
    @pytest.mark.parametrize("occupancy", OCCUPANCY_VALUES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZE_VALUES)
    def test_forward_pass_correctness(self, n_in, n_out, block_size, occupancy, batch_size):
        """Test that SparseLinear and emulator produce identical forward outputs."""
        device = "cuda"
        dtype = torch.float32
        
        # Create layers
        sparse_layer = SparseLinear(n_in, n_out, block_size, occupancy, device, dtype)
        emulator_layer = SparseLinearEmulater(sparse_layer)
        
        # Test input
        x = torch.randn(batch_size, n_in, device=device, dtype=dtype)
        
        # Forward passes
        y_sparse = sparse_layer(x)
        y_emulator = emulator_layer(x)
        
        # Assertions
        assert y_sparse.shape == y_emulator.shape, "Output shapes must match"
        assert y_sparse.abs().max().item() > 1, "Sparse output should be non-trivial"
        assert y_emulator.abs().max().item() > 1, "Emulator output should be non-trivial"
        assert torch.allclose(y_sparse, y_emulator, atol=1e-2, rtol=1e-2), "Outputs should be close"
        
        # Additional check for max difference
        max_diff = (y_sparse - y_emulator).abs().max().item()
        assert max_diff < 0.01, f"Max difference {max_diff} should be small"
    
    @pytest.mark.parametrize("n_in", N_IN_VALUES)
    @pytest.mark.parametrize("n_out", N_OUT_VALUES)
    @pytest.mark.parametrize("block_size", BLOCK_SIZE_VALUES)
    @pytest.mark.parametrize("occupancy", OCCUPANCY_VALUES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZE_VALUES)
    def test_backward_pass_correctness(self, n_in, n_out, block_size, occupancy, batch_size):
        """Test that SparseLinear and emulator produce identical backward gradients."""
        device = "cuda"
        dtype = torch.float32
        
        # Create layers
        sparse_layer = SparseLinear(n_in, n_out, block_size, occupancy, device, dtype)
        emulator_layer = SparseLinearEmulater(sparse_layer)
        
        # Test inputs (with gradients)
        x_sparse = torch.randn(batch_size, n_in, device=device, dtype=dtype, requires_grad=True)
        x_emulator = x_sparse.detach().clone().requires_grad_(True)
        
        # Forward passes
        y_sparse = sparse_layer(x_sparse)
        y_emulator = emulator_layer(x_emulator)
        
        # Backward passes
        loss_sparse = y_sparse.sum()
        loss_emulator = y_emulator.sum()
        loss_sparse.backward()
        loss_emulator.backward()
        
        # Assertions
        assert x_sparse.grad.shape == x_emulator.grad.shape, "Input gradient shapes must match"
        assert x_sparse.grad.abs().max().item() > 1, "Sparse input gradients should be non-trivial"
        assert x_emulator.grad.abs().max().item() > 1, "Emulator input gradients should be non-trivial"
        assert torch.allclose(x_sparse.grad, x_emulator.grad, atol=1e-2, rtol=1e-2), "Input gradients should be close"
        
        # Additional check for gradient difference
        grad_diff = (x_sparse.grad - x_emulator.grad).abs().max().item()
        assert grad_diff < 0.01, f"Input gradient max difference {grad_diff} should be small"
    
    @pytest.mark.parametrize("n_in", N_IN_VALUES)
    @pytest.mark.parametrize("n_out", N_OUT_VALUES)
    @pytest.mark.parametrize("block_size", BLOCK_SIZE_VALUES)
    @pytest.mark.parametrize("occupancy", OCCUPANCY_VALUES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZE_VALUES)
    def test_optimization_behavior(self, n_in, n_out, block_size, occupancy, batch_size):
        """Test that SparseLinear and emulator have identical optimization behavior."""
        device = "cuda"
        dtype = torch.float32
        
        # Create layers
        sparse_layer = SparseLinear(n_in, n_out, block_size, occupancy, device, dtype)
        emulator_layer = SparseLinearEmulater(sparse_layer)
        
        # Create optimizers
        optimizer_sparse = torch.optim.SGD(sparse_layer.parameters(), lr=0.1)
        optimizer_emulator = torch.optim.SGD(emulator_layer.parameters(), lr=0.1)
        
        # Test input and target
        x = torch.randn(batch_size, n_in, device=device, dtype=dtype)
        target = torch.zeros(batch_size, n_out, device=device, dtype=dtype)
        
        # Check initial weights are identical
        initial_sparse_norm = sparse_layer.weight.norm().item()
        initial_emulator_norm = emulator_layer.matrix.norm().item()
        assert abs(initial_sparse_norm - initial_emulator_norm) < 1e-5, "Initial weights should be identical"
        
        # Run optimization steps
        for step in range(3):
            # Sparse layer update
            optimizer_sparse.zero_grad()
            y_sparse = sparse_layer(x)
            loss_sparse = torch.nn.functional.mse_loss(y_sparse, target)
            loss_sparse.backward()
            optimizer_sparse.step()
            
            # Emulator layer update
            optimizer_emulator.zero_grad()
            y_emulator = emulator_layer(x)
            loss_emulator = torch.nn.functional.mse_loss(y_emulator, target)
            loss_emulator.backward()
            optimizer_emulator.step()
            
            # Check weight norms are close after each step
            sparse_norm = sparse_layer.weight.norm().item()
            emulator_norm = emulator_layer.matrix.norm().item()
            norm_diff = abs(sparse_norm - emulator_norm)
            
            assert norm_diff < 1e-4, f"Step {step+1}: Weight norm difference {norm_diff} should be tiny"
            assert sparse_norm < initial_sparse_norm, f"Step {step+1}: Sparse weights should decrease"
            assert emulator_norm < initial_emulator_norm, f"Step {step+1}: Emulator weights should decrease"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])