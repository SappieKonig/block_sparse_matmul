import torch
import triton
import triton.language as tl
import time


@triton.jit
def column_sparse(a_ptr, b_ptr, c_ptr):
    ...


@triton.jit
def sparse_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    
    m_mask = offs_m < M
    n_mask = offs_n < N
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = offs_k < K
        
        a_ptrs = a_ptr + (offs_k[None, :] * stride_ak + offs_m[:, None] * stride_am)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        
        a = tl.load(a_ptrs, mask=k_mask[None, :] & m_mask[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        accumulator += tl.dot(a, b)
    
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tl.store(c_ptrs, accumulator, mask=m_mask[:, None] & n_mask[None, :])

def triton_block_sparse_matmul(A_dense, B, block_size=128):
    M, K = B.shape
    K_A, N = A_dense.shape
    assert K == K_A, f"Dimension mismatch: B.shape[1]={K} != A.shape[0]={K_A}"
    
    # Transpose for correct GEMM: C = B @ A
    C = torch.zeros((M, N), dtype=torch.float32, device=A_dense.device)
    
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 32, 32, 32
    grid = ((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    
    sparse_matmul_kernel[grid](
        B, A_dense, C,
        M, N, K,
        B.stride(0), B.stride(1),
        A_dense.stride(0), A_dense.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return C.to(A_dense.dtype)

def bench_triton(B, A, block_size=128, iters=20):
    triton_block_sparse_matmul(A, B, block_size)
    if B.device.type == "cuda":
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters): _ = triton_block_sparse_matmul(A, B, block_size)
        t1.record(); torch.cuda.synchronize()
        return t0.elapsed_time(t1) / iters
    else:
        t0 = time.perf_counter()
        for _ in range(iters): _ = triton_block_sparse_matmul(A, B, block_size)
        return (time.perf_counter() - t0) * 1e3 / iters