import torch
import random

torch.ops.load_library("build/libtiledcuda.so")


if __name__ == "__main__":
    a = torch.randn(256, 256, device='cuda')
    b = torch.randn(256, 256, device='cuda')
    c = torch.empty(256, 256, device='cuda')
    a_data = a.flatten().half()
    b_data = b.flatten().half()
    c_data = c.flatten().half()

    torch.ops.tiledcuda.gemm(a_data, b_data, c_data, 256, 256, 256)
    ref_c = torch.mm(a.half(), b.half().t()).flatten()

    print(c_data)
    print(ref_c)

    assert torch.allclose(c_data, ref_c, atol=1e-3)
