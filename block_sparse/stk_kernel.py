import time
import torch
import stk

def bench_stk(B, A_sparse, iters=20):
    if B.device.type == "cuda":
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters): _ = stk.ops.dds(B, A_sparse)            # dense @ sparse
        t1.record(); torch.cuda.synchronize()
        return t0.elapsed_time(t1) / iters
    else:
        t0 = time.perf_counter()
        for _ in range(iters): _ = stk.ops.dds(B, A_sparse)
        return (time.perf_counter() - t0) * 1e3 / iters