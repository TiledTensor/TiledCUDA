import unittest
import torch

import context

from pytiledcuda import batched_gemm


class TestBatchedGemm(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1234)

    def test_batched_gemm(self):
        m = 256
        n = 256
        k = 256
        batch_count = 10

        a = torch.randn(batch_count, m, k, device='cuda')
        b = torch.randn(batch_count, k, n, device='cuda')
        c = torch.empty(batch_count, m, n, device='cuda')

        a_data = a.flatten().half()
        b_data = b.transpose(1, 2).flatten().half()
        c_data = c.flatten().half()

        batched_gemm(a_data, b_data, c_data, m, n, k, batch_count)
        ref_c = torch.bmm(a.half().view(batch_count, m, k),
                          b.half().view(batch_count, k, n)).flatten()

        print(c_data)
        print(ref_c)

        assert torch.allclose(c_data, ref_c, atol=1e-3)


if __name__ == "__main__":

    unittest.main()
