import torch
from torch import Tensor

from gemm import gemm_func


def run_unittest(a: Tensor, b: Tensor, c: Tensor, M, N, K, TM, TN):
    gemm_func(a, b, c, M, N, K, TM, TN)
    print("Result:")
    print(c)

    print("\nReference:")
    ref_c = a @ b.t()
    print(ref_c)

    if not torch.allclose(c.half(), ref_c, atol=1e-3):
        return False
    else:
        return True


def run_test(M: int, N: int, K: int, TM: int, TN: int):
    device = torch.device("cuda")
    dtype = torch.float16

    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)
    c = torch.zeros(M, N, device=device, dtype=torch.float32)

    if not run_unittest(a, b, c, M, N, K, TM, TN):
        raise RuntimeError("Failed unittest.")
    print("Passed unittest.")

    for i in range(5):  # warm up
        gemm_func(a, b, c, M, N, K, TM, TN)
        ref_c = a @ b.t()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    iters = 50

    start_event.record()
    for i in range(iters):
        gemm_func(a, b, c, M, N, K, TM, TN)
    end_event.record()
    torch.cuda.synchronize()

    time1 = start_event.elapsed_time(end_event) / iters

    start_event.record()
    for i in range(iters):
        ref_c = a @ b.t()
    end_event.record()
    torch.cuda.synchronize()

    time2 = start_event.elapsed_time(end_event) / iters
    return time1, time2


if __name__ == "__main__":
    M = 4096
    N = 4096
    K = 4096
    time1, time2 = run_test(M, N, K, 256, 128)
    print("Shape\ttiledcuda(ms)\tcublass(ms)\tRatio")
    print("[{}, {}, {}]\t{:.4f}\t{:.4f}\t{:.3f}".format(
        M, N, K, time1, time2, time1 / time2))
