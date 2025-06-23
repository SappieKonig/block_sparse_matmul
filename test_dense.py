import torch
import time


N = 2 ** 11

A = torch.randn((N, N), dtype=torch.float16, device="cuda")
B = torch.randn((N, N), dtype=torch.float16, device="cuda")

print("Warming up...")
for _ in range(10):
    C = torch.mm(A, B)
    torch.cuda.synchronize()

times = []
for _ in range(50):
    start = time.perf_counter()
    C = torch.mm(A, B)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)

print(f"Time: {sum(times) / len(times):.2f} ms")
tflops = N * N * N * 2 / (sum(times) / len(times)) / 1e12
print(f"TFLOPS: {tflops:.2f}")
