import unittest

import torch

import context
from pytiledcuda import back2back_gemm


class Back2BackGemm(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1234)

    def test_b2b_gemm(self):

        device = torch.device("cuda")
        dtype = torch.float16

        a = torch.randn(256, 256, device=device, dtype=dtype)
        b = torch.randn(256, 256, device=device, dtype=dtype)
        c = torch.randn(256, 256, device=device, dtype=dtype)
        d = torch.empty(256, 256, device=device, dtype=dtype)

        back2back_gemm(a, b, c, d, 256, 256, 256, 256)

        ref_d = (a @ b.t()) @ c.t()
        print(d)
        print(ref_d)

        assert torch.allclose(d, ref_d, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
