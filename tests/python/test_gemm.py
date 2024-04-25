import torch
import sys

import unittest

sys.path.append('./')


class TestGemm(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1234)

    def test_gemm(self):
        import pytiledcuda

        a = torch.randn(256, 256, device='cuda')
        b = torch.randn(256, 256, device='cuda')
        c = torch.empty(256, 256, device='cuda')

        a_data = a.flatten().half()
        b_data = b.flatten().half()
        c_data = c.flatten().half()

        pytiledcuda.gemm(a_data, b_data, c_data, 256, 256, 256)
        ref_c = torch.mm(a.half(), b.half().t()).flatten()

        self.assertTrue(torch.allclose(c_data, ref_c, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
