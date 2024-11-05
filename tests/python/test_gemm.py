import unittest

import torch

import context
from pytiledcuda import gemm


class TestGemm(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1234)

    def test_gemm(self):
        device = torch.device("cuda")
        dtype = torch.float16

        M = 4096
        N = 4096
        K = 2048

        a = torch.randn(M, K, device=device, dtype=dtype)
        b = torch.randn(N, K, device=device, dtype=dtype)
        c = torch.zeros(M, N, device=device, dtype=dtype)

        gemm(a, b, c, M, N, K)
        ref_c = a @ b.t()

        print("c: ", c)
        print("ref_c: ", ref_c)

        epsilon = 2e-5

        avg_diff = (torch.sum(torch.abs(ref_c - c) / (M * N))).item()

        print('avg_diff: ', avg_diff)

        assert avg_diff < epsilon



if __name__ == "__main__":
    unittest.main()
