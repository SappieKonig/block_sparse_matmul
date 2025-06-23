import torch
import time
import triton
import triton.language as tl
from tqdm import tqdm
import random


# for sparse kernel of A @ B, where only B is sparse

# PARAMETERS
M = 2 ** 11
K = 2 ** 11
N = 2 ** 11
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64
OCCUPANCY = 4

TOTAL_K_BLOCKS = K // BLOCK_K
TOTAL_N_BLOCKS = N // BLOCK_N
NUM_TRIALS = 50
NUM_WARMUP = 10

@triton.jit
def block_sparse_mm_kernel(
        A_ptr, B_data_ptr, B_indices_ptr, C_ptr,
        block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr, occupancy: tl.constexpr,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_bi_o, stride_bi_n):

    pid_m = tl.program_id(0)  # batch block
    pid_n = tl.program_id(1)  # output column block
    
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    
    for i in range(occupancy):
        block_idx = pid_n * stride_bi_n + i * stride_bi_o
        row_block_idx = tl.load(B_indices_ptr + block_idx)
        
        # Load A tile
        offs_k = row_block_idx * block_k + tl.arange(0, block_k)
        A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_tile = tl.load(A_ptrs)
        
        # Load B tile
        data_col_start = (row_block_idx * occupancy + i) * block_n
        offs_b_k = tl.arange(0, block_k)
        offs_b_n = data_col_start + tl.arange(0, block_n)
        B_ptrs = B_data_ptr + offs_b_k[:, None] * stride_bk + offs_b_n[None, :] * stride_bn
        b_tile = tl.load(B_ptrs)
        
        acc += tl.dot(a_tile, b_tile)
    
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc.to(tl.float16))

def block_sparse_mm(A, B_data, B_indices):
    C = torch.zeros((M, N), dtype=A.dtype, device=A.device)
    
    grid = (M // BLOCK_M, N // BLOCK_N)
    
    block_sparse_mm_kernel[grid](
        A, B_data, B_indices, C,
        BLOCK_M, BLOCK_N, BLOCK_K, OCCUPANCY,
        A.stride(0), A.stride(1),
        B_data.stride(0), B_data.stride(1),
        C.stride(0), C.stride(1),
        B_indices.stride(0), B_indices.stride(1)
    )
    
    return C


def construct_sparse_matrix(k, n, block_k, block_n, occupancy, total_n_blocks, device, dtype):
    B_data = torch.randn((block_k, block_n * occupancy * total_n_blocks), dtype=dtype, device=device)
    B_indices = []
    for _ in range(occupancy):
        while True:
            indices = list(range(total_n_blocks))
            random.shuffle(indices)
            for ind in B_indices:
                if any(x == y for x, y in zip(ind, indices)):
                    break
            else:
                B_indices.append(indices)
                break

    B = torch.zeros((k, n), dtype=dtype, device=device)
    indices = torch.zeros((total_n_blocks, occupancy), dtype=torch.int32, device=device)
    for col in range(total_n_blocks):
        for i in range(occupancy):
            block_idx = col * occupancy + i
            row = B_indices[i][col]
            indices[col, i] = row
            
            row_start = row * block_k
            col_start = col * block_n
            data_start = block_idx * block_n
            
            B[row_start:row_start + block_k, col_start:col_start + block_n] = \
                B_data[:, data_start:data_start + block_n]

    return B_data, indices, B


def main():
    device = "cuda"
    dtype = torch.float16

    # quick check so I remember how I mapped the indices
    B_data, B_indices, B = construct_sparse_matrix(k=4, n=4, block_k=1, block_n=1, occupancy=2, total_n_blocks=4, device=device, dtype=dtype)
    print(*B_data.tolist(), sep="\n", end="\n\n")
    print(*B_indices.tolist(), sep="\n", end="\n\n")
    print(*B.tolist(), sep="\n", end="\n")

    print(f"Sparsity: {OCCUPANCY}/{TOTAL_K_BLOCKS} = {OCCUPANCY/TOTAL_K_BLOCKS:.1%}")
    

    B_data, B_indices, B = construct_sparse_matrix(k=M, n=N, block_k=BLOCK_K, block_n=BLOCK_N, occupancy=OCCUPANCY, total_n_blocks=TOTAL_N_BLOCKS, device=device, dtype=dtype)

    A = torch.randn((M, K), dtype=dtype, device=device)
    
    # Warmup
    print("Warming up...")
    for _ in range(NUM_WARMUP):
        _ = block_sparse_mm(A, B_data, B_indices)
        _ = torch.mm(A, B)
        torch.cuda.synchronize()
    
    # Benchmark sparse
    torch.cuda.synchronize()
    start = time.time()
    for _ in tqdm(range(NUM_TRIALS), desc="Sparse"):
        C_sparse = block_sparse_mm(A, B_data, B_indices)
        torch.cuda.synchronize()
    sparse_time = (time.time() - start) / NUM_TRIALS * 1000
    
    # Benchmark dense
    torch.cuda.synchronize()
    start = time.time()
    for _ in tqdm(range(NUM_TRIALS), desc="Dense"):
        C_dense = torch.mm(A, B)
        torch.cuda.synchronize()
    dense_time = (time.time() - start) / NUM_TRIALS * 1000
    
    # Check correctness
    C_sparse_check = block_sparse_mm(A, B_data, B_indices)
    C_dense_check = torch.mm(A, B)
    max_diff = torch.max(torch.abs(C_sparse_check - C_dense_check)).item()
    
    print(f"\nResults:")
    print(f"Sparse: {sparse_time:.2f} ms")
    print(f"Dense:  {dense_time:.2f} ms") 
    print(f"Speedup: {dense_time/sparse_time:.2f}x")
    print(f"Max diff: {max_diff:.6f}")
    print(f"Correct: {'YES' if max_diff < 1e-2 else 'NO'}")

if __name__ == "__main__":
    main()