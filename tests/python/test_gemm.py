import torch
import random

torch.ops.load_library("build/libtiledcuda.so")


if __name__ == "__main__":
    a = torch.randn(256, 256, device='cuda')
    b = torch.randn(256, 256, device='cuda')
    c = torch.empty(256, 256, device='cuda')
    a_data = a.flatten()
    b_data = b.flatten()
    c_data = c.flatten()
    torch.ops.tiledcuda.gemm(a, b, c, 256, 256, 256)

    ref_c = torch.mm(a, b)

    print(c)
    print(ref_c)
