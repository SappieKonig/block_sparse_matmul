import torch
from layers import SparseLinear, SparseLinearEmulater

# Simple equality check between SparseLinear and SparseLinearEmulator
device = "cuda"
dtype = torch.float32
n_in, n_out = 1024, 512
block_size, occupancy = 64, 4
batch_size = 128

# Create SparseLinear
sparse_layer = SparseLinear(n_in, n_out, block_size, occupancy, device, dtype)

# Create emulator from sparse layer
emulator_layer = SparseLinearEmulater(sparse_layer)

# Test input (with gradients)
x_sparse = torch.randn(batch_size, n_in, device=device, dtype=dtype, requires_grad=True)
x_emulator = x_sparse.detach().clone().requires_grad_(True)

# Forward passes
y_sparse = sparse_layer(x_sparse)
y_emulator = emulator_layer(x_emulator)

# Check forward equality
print(f"=== Forward Pass ===")
print(f"Output shapes match: {y_sparse.shape == y_emulator.shape}")
print(f"Sparse max output: {y_sparse.abs().max().item():.6f}")
print(f"Emulator max output: {y_emulator.abs().max().item():.6f}")
print(f"Outputs close: {torch.allclose(y_sparse, y_emulator, atol=1e-3, rtol=1e-2)}")
print(f"Max difference: {(y_sparse - y_emulator).abs().max().item():.6f}")

# Backward passes
loss_sparse = y_sparse.sum()
loss_emulator = y_emulator.sum()
loss_sparse.backward()
loss_emulator.backward()

# Check backward equality
print(f"\n=== Backward Pass ===")
print(f"Input grad shapes match: {x_sparse.grad.shape == x_emulator.grad.shape}")
print(f"Sparse input grad max: {x_sparse.grad.abs().max().item():.6f}")
print(f"Emulator input grad max: {x_emulator.grad.abs().max().item():.6f}")
print(f"Input grads close: {torch.allclose(x_sparse.grad, x_emulator.grad, atol=1e-3, rtol=1e-2)}")
print(f"Input grad max difference: {(x_sparse.grad - x_emulator.grad).abs().max().item():.6f}")

# Optimization test
print(f"\n=== Optimization Test ===")
optimizer_sparse = torch.optim.SGD(sparse_layer.parameters(), lr=0.1)
optimizer_emulator = torch.optim.SGD(emulator_layer.parameters(), lr=0.1)

# Target output (zeros to drive weights to zero)
target = torch.zeros_like(y_sparse)

print(f"Initial sparse weight norm: {sparse_layer.weight.norm().item():.6f}")
print(f"Initial emulator weight norm: {emulator_layer.matrix.norm().item():.6f}")

for step in range(5):
    # Sparse layer update
    optimizer_sparse.zero_grad()
    y_sparse = sparse_layer(x_sparse.detach())
    loss_sparse = torch.nn.functional.mse_loss(y_sparse, target)
    loss_sparse.backward()
    optimizer_sparse.step()
    
    # Emulator layer update  
    optimizer_emulator.zero_grad()
    y_emulator = emulator_layer(x_emulator.detach())
    loss_emulator = torch.nn.functional.mse_loss(y_emulator, target)
    loss_emulator.backward()
    optimizer_emulator.step()
    
    print(f"Step {step+1}: Sparse norm={sparse_layer.weight.norm().item():.6f}, Emulator norm={emulator_layer.matrix.norm().item():.6f}")