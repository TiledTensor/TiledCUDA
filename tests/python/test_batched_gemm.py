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

        device = torch.device("cuda")
        dtype = torch.float16

        a = torch.randn(batch_count, m, k, device=device, dtype=dtype)
        b = torch.randn(batch_count, k, n, device=device, dtype=dtype)
        c = torch.empty(batch_count, m, n, device=device, dtype=dtype)

        b = b.transpose(1, 2)

        batched_gemm(a, b, c, m, n, k, batch_count)
        ref_c = torch.bmm(a.view(batch_count, m, k), b.view(batch_count, k, n))

        print(c)
        print(ref_c)

        assert torch.allclose(c, ref_c, atol=1e-3)


if __name__ == "__main__":

    unittest.main()
