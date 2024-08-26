import unittest

import torch

import context
# from pytiledcuda import flashattention

class TestFlashAttention(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1234)

    def test_flash_attention(self):
        m = 256
        n = 256
        k = 256
        batch_count = 1

        Q = torch.randn(m, k, device='cuda')
        K = torch.randn(k, n, device='cuda')
        V = torch.randn(n, k, device='cuda')
        out = torch.empty( m, n, device='cuda')

        ref_o = torch.nn.functional.scaled_dot_product_attention(Q.half(), K.half(), V.half())

        print(ref_o)

if __name__ == "__main__":

    unittest.main()

    