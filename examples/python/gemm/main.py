import torch

from gemm import gemm_func

if __name__ == "__main__":
    device = torch.device("cuda")
    dtype = torch.float16

    M = 128
    N = 128
    K = 256

    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)
    c = torch.zeros(M, N, device=device, dtype=torch.float32)

    gemm_func(a, b, c, M, N, K)

    print("Result:")
    print(c)

    print("\nReference:")
    ref_c = a @ b.t()
    print(ref_c)

    if not torch.allclose(c.half(), ref_c, atol=1e-3):
        raise RuntimeError("test failed")
    else:
        print("test passed")
