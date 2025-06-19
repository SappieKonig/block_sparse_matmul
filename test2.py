import torch
from random import shuffle
import triton
import triton.language as tl


def generate_column_block_sparse_matrix(block_size: int, num_blocks: int, occupancy: int, device: str, dtype: torch.dtype):
    coordinates = []
    for _ in range(num_blocks):
        indices = list(range(num_blocks))
        shuffle(indices)
        coordinates.append(sorted(indices[:occupancy]))

    data = torch.randn((occupancy * block_size, num_blocks * block_size), dtype=dtype, device=device)
    return data, coordinates


def full_tensor(data, coordinates, block_size):
    """
    Convert column block sparse format to full dense tensor.
    
    Args:
        data: Compressed sparse data tensor (occupancy*block_size, num_blocks*block_size)
        coordinates: List of lists containing block row indices for each column
        block_size: Size of each block
    
    Returns:
        Dense tensor representation
    """
    rows, cols = data.shape
    num_blocks = cols // block_size
    occupancy = rows // block_size
    
    base_tensor = torch.zeros((num_blocks * block_size, num_blocks * block_size), 
                            dtype=data.dtype, device=data.device)
    
    for j in range(num_blocks):
        for i in range(occupancy):
            data_slice = data[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            row_start = block_size * coordinates[j][i]
            base_tensor[row_start:row_start+block_size, j*block_size:(j+1)*block_size] = data_slice
    
    return base_tensor


def torch_blockwise_matmul(data, coordinates, B, block_size):
    occupancy_bs, num_blocks_bs = data.shape
    occupancy = occupancy_bs // block_size
    num_blocks = num_blocks_bs // block_size
    
    bs, _ = B.shape
    M = num_blocks * block_size
    
    C = torch.zeros((bs, M), dtype=data.dtype, device=data.device)

    for i in range(num_blocks):
        accum = torch.zeros((bs, block_size), dtype=data.dtype, device=data.device)
        
        for j in range(occupancy):
            coordinate = coordinates[i][j]
            data_block = data[j*block_size:(j+1)*block_size, i*block_size:(i+1)*block_size]
            B_block = B[:, coordinate*block_size:(coordinate+1)*block_size]
            accum += B_block @ data_block
        C[:, i*block_size:(i+1)*block_size] = accum

    return C


BLOCK = 128          # tile size in every dimension


@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr,
                  M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK: tl.constexpr):
    """
    C[M, N] = A[M, K] @ B[K, N]   with BLOCK = 128, dim sizes % 128 == 0
    """
    # ------------------------------------------------------------------
    # identify the 128×128 tile this program instance is responsible for
    pid_m = tl.program_id(0)          # row-tile index
    pid_n = tl.program_id(1)          # col-tile index

    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)        # 0…127 along M
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)        # 0…127 along N

    # accumulator for the current C tile
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)

    # loop over the K dimension in 128-wide strips
    for k in range(0, K, BLOCK):
        offs_k = k + tl.arange(0, BLOCK)                # 0…127 along K

        # pointers to the current 128×128 sub-matrices
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # load tiles (no boundary checks needed — sizes are multiples of 128)
        a = tl.load(a_ptrs)        # (128, 128)
        b = tl.load(b_ptrs)        # (128, 128)

        # multiply-accumulate
        acc += tl.dot(a, b)        # (128, 128)

    # write the resulting C tile
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)
    

def triton_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Minimal wrapper: C = a @ b for sizes that are multiples of 128."""
    assert a.ndim == b.ndim == 2
    M, K = a.shape
    K2, N = b.shape
    assert K == K2,  "inner dimensions must match"
    assert (M % BLOCK == N % BLOCK == K % BLOCK == 0), "sizes must be multiples of 128"
    assert a.is_cuda and b.is_cuda, "tensors must be on the same CUDA device"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # one Triton program instance (= CTA) per 128×128 output tile
    grid = (M // BLOCK, N // BLOCK)

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK=BLOCK,
        num_warps=4,          # modest default; change if you like
        num_stages=1          # “optimization-less” :)
    )
    return c


BLOCK      = 128            # tile width/height
OCCUPANCY  = 4              # <-- keep this a compile-time constant

###############################################################################
#  Kernel
###############################################################################
@triton.jit
def col_block_sparse_mm_kernel(
        B_ptr,              #  float16 [Bs,  num_blocks*BLOCK]
        D_ptr,              #  float16 [OCCUPANCY*BLOCK, num_blocks*BLOCK]  (sparse data)
        COORD_ptr,          #     int32 [num_blocks, OCCUPANCY]              (row-block indices)
        C_ptr,              #  float16 [Bs,  num_blocks*BLOCK]

        Bs, num_blocks,                                   # batch size, #block-columns
        stride_bm, stride_bk,                             # B  strides
        stride_dm, stride_dn,                             # D  strides
        stride_cm, stride_cn,                             # C  strides
        BLOCK: tl.constexpr,
        OCC:   tl.constexpr):                             # OCCUPANCY (compile-time)

    ############  identify which 128×128 tile we own  ############
    pid_m = tl.program_id(0)            # batch-block (rows of C/B)
    pid_n = tl.program_id(1)            # column-block (cols of C / block-columns of A)

    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)          # [0..127] in batch dim
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)          # [0..127] in output-col dim

    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)      # FP32 accumulator

    ###########################################################################
    # loop over *only* the non-zero blocks that feed this column-block
    ###########################################################################
    # ptr to the list of OCC integers telling us which row-blocks to read
    coord_base = COORD_ptr + pid_n * OCC

    for j in tl.static_range(OCC):                        # unrolled, compile-time
        coord = tl.load(coord_base + j)                   # scalar int32, value ∈ [0, num_blocks)

        ############################  B-tile  ############################
        offs_k = coord * BLOCK + tl.arange(0, BLOCK)      # columns of B to read
        B_ptrs = (B_ptr
                  + offs_m[:, None] * stride_bm           # rows   (batch)
                  + offs_k[None, :] * stride_bk)          # cols   (K)
        b_tile = tl.load(B_ptrs)                          # (128, 128)  float16

        ############################  A-tile (= D-block)  ###############
        d_row = j * BLOCK + tl.arange(0, BLOCK)           # rows inside D
        d_col = pid_n * BLOCK + tl.arange(0, BLOCK)       # cols inside D
        D_ptrs = (D_ptr
                  + d_row[:, None] * stride_dm
                  + d_col[None, :] * stride_dn)
        a_tile = tl.load(D_ptrs)                          # (128, 128)  float16

        ############################  FMA  ###############################
        acc += tl.dot(b_tile, a_tile)                     # FP32 accumulate

    #########################  write out C-tile  ########################
    C_ptrs = (C_ptr
              + offs_m[:, None] * stride_cm
              + offs_n[None, :] * stride_cn)
    tl.store(C_ptrs, acc.to(tl.float16))                 # cast back to fp16
    

def sparse_triton_mm(B: torch.Tensor,
                     data: torch.Tensor,
                     coords: torch.Tensor  # int32, shape [num_blocks, OCCUPANCY]
                     ) -> torch.Tensor:
    """
    B  – [batch_size, num_blocks*128]           (dense)
    A  – column-block sparse given by (data, coords)
    returns C = B @ A   with      BLOCK = 128, OCCUPANCY compile-time constant
    """
    assert B.dtype == data.dtype == torch.float16
    assert B.is_cuda  and data.is_cuda and coords.is_cuda
    batch_size, K = B.shape
    num_blocks = K // BLOCK
    assert coords.shape == (num_blocks, OCCUPANCY)

    C = torch.empty_like(B)

    grid = (batch_size // BLOCK, num_blocks)   # one CTA per 128×128 output tile
    col_block_sparse_mm_kernel[grid](
        B, data, coords, C,
        batch_size, num_blocks,
        B.stride(0), B.stride(1),
        data.stride(0), data.stride(1),
        C.stride(0), C.stride(1),
        BLOCK=BLOCK,
        OCC=OCCUPANCY,
        num_warps=4,
        num_stages=1
    )
    return C



if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float16
    block_size = 128
    num_blocks = 16
    occupancy = 4
    batch_size = 512

    B = torch.randn((batch_size, num_blocks*block_size), dtype=dtype, device=device)

    data, coordinates = generate_column_block_sparse_matrix(block_size, num_blocks, occupancy, device, dtype)
    A_full = full_tensor(data, coordinates, block_size)
    C_ref = B @ A_full
    C_blockwise = torch_blockwise_matmul(data, coordinates, B, block_size)
    C_triton = triton_mm(B, A_full)
    torch_coords = torch.tensor(coordinates, dtype=torch.int32, device=device)
    C_triton_sparse = sparse_triton_mm(B, data, torch_coords)

    print(C_ref.max())
    print(C_blockwise.max())
    print(C_triton.max())
    print(C_triton_sparse.max())
    
    print((C_ref - C_blockwise).abs().max())
    
    
