import unittest

import torch

import context
from pytiledcuda import back2back_gemm


class Back2BackGemm(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1234)

    def test_b2b_gemm(self):

        a = torch.randn(256, 256, device='cuda')
        b = torch.randn(256, 256, device='cuda')
        c = torch.randn(256, 256, device='cuda')
        d = torch.empty(256, 256, device='cuda')

        a_data = a.flatten().half()
        b_data = b.flatten().half()
        c_data = c.flatten().half()
        d_data = d.flatten().half()

        back2back_gemm(a_data, b_data, c_data, d_data, 256, 256, 256, 256)

        ref_d = torch.mm(a.half(), b.half().t()).mm(c.half().t()).flatten()
        print(d_data)
        print(ref_d)

        assert torch.allclose(d_data, ref_d, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
