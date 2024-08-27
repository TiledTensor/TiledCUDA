import unittest

import torch

import context
from pytiledcuda import flash_attention_fwd

class TestFlashAttention(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1234)

    def test_flash_attention(self):
        batch_size = 1
        num_heads = 128
        embed_dim = 128

        Q = torch.randn(num_heads, embed_dim, device='cuda')
        K = torch.randn(num_heads, embed_dim, device='cuda')
        V = torch.randn(num_heads, embed_dim, device='cuda')
        O = torch.empty(num_heads, embed_dim, device='cuda')

        # O = softmax(Q @ K.T) @ V
        ref_o = torch.nn.functional.scaled_dot_product_attention(Q.half(), K.half(), V.half())

        Q_data = Q.flatten().half()
        K_data = K.flatten().half()
        V_data = V.t().flatten().half()
        O_data = O.flatten().half()

        flash_attention_fwd(Q_data, K_data, V_data, O_data, batch_size, num_heads, embed_dim)

        print(ref_o)

        ans_o = O_data.view(num_heads, embed_dim).half()
        print(ans_o)

if __name__ == "__main__":

    unittest.main()

    