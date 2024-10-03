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

        M = 256
        N = 256
        K = 128

        a = torch.randn(M, K, device=device, dtype=dtype)
        b = torch.randn(N, K, device=device, dtype=dtype)
        c = torch.empty(M, N, device=device, dtype=dtype)

        gemm(a, b, c, M, N, K)
        ref_c = torch.mm(a, b.t())

        self.assertTrue(torch.allclose(c, ref_c, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
