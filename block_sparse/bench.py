import time, torch
from random import shuffle
import stk
from triton_kernel import triton_block_sparse_matmul, bench_triton
from stk_kernel import bench_stk

# ─── helper ---------------------------------------------------------------
def make_block_sparse(num_blocks, block_size, fill, device, dtype):
    """Return an (num_blocks·block_size)² tensor with `fill` non-zero blocks."""
    A = torch.zeros((num_blocks*block_size, num_blocks*block_size), dtype=dtype, device=device)
    idx = list(range(num_blocks * num_blocks))
    shuffle(idx)
    for flat in idx[:fill]:
        i, j = divmod(flat, num_blocks)
        xs, ys = i*block_size, j*block_size
        A[xs:xs+block_size, ys:ys+block_size] = torch.randn(
            (block_size, block_size), dtype=dtype, device=device
        )
    return A


def bench_dense(B, A, iters=20):
    if B.device.type == "cuda":
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters): _ = B @ A
        t1.record(); torch.cuda.synchronize()
        return t0.elapsed_time(t1) / iters          # ms
    else:
        t0 = time.perf_counter()
        for _ in range(iters): _ = B @ A
        return (time.perf_counter() - t0) * 1e3 / iters

# ─── main ----------------------------------------------------------------
if __name__ == "__main__":
    dev   = "cuda"
    dtype = torch.bfloat16
    batch_size, num_blocks, block_size, fill = 8192, 64, 128, 128

    A = make_block_sparse(num_blocks, block_size, fill, dev, dtype)
    B = torch.randn((batch_size, num_blocks*block_size), dtype=dtype, device=dev)

    print(A.shape, B.shape)

    dense_ms = bench_dense(B, A)

    import stk
    A_sp   = stk.ops.to_sparse(A, blocking=block_size)            # tensor → stk.Matrix
    stk_ms = bench_stk(B, A_sp)
    stk_diff   = (B @ A).float().sub(stk.ops.dds(B, A_sp).float()).abs().max().item()

    triton_ms = bench_triton(B, A, block_size)
    triton_diff = (B @ A).float().sub(triton_block_sparse_matmul(A, B, block_size).float()).abs().max().item()

    print(f"Device              : {dev}")
    print(f"Dense GEMM          : {dense_ms:.4f}  ms")
    print(f"STK dense@sparse   : {stk_ms:.4f}  ms   (speed-up ×{dense_ms/stk_ms:.2f})")
    print(f"Triton sparse      : {triton_ms:.4f}  ms   (speed-up ×{dense_ms/triton_ms:.2f})")
    print(f"STK max ‖Δ‖_∞       : {stk_diff:e}")
    print(f"Triton max ‖Δ‖_∞    : {triton_diff:e}")
