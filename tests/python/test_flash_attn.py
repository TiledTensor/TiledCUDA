import unittest

import torch

import context
from pytiledcuda import flash_attention_fwd

class FlashAttention:

    def __init__(self, Q, K, V, O, m, n, k, p, ktm, ktn, ktk, ktp):
        self.m = m
        self.n = n
        self.k = k
        self.p = p
        self.ktm = ktm
        self.ktn = ktn
        self.ktk = ktk
        self.ktp = ktp

        self.Q = Q # m * k
        self.K = K # n * k
        self.V = V # n * p
        self.O = O # m * p

    
    def attn_func(self, prev_maxes, prev_sums, prev_out, q, k, v):
        attn_weights = q @ k.T

        # reduce maxes
        cur_maxes = torch.max(attn_weights, dim=-1, keepdim=True)
        exp_weights = torch.exp(attn_weights - cur_maxes)
        # unnormalized attention score @ values
        exp_values = exp_weights @ v
        # move the normalization step to the very end of the attention computation.
        cur_sums = torch.sum(exp_weights, dim=-1, keepdim=True) # l(x_cur)

        # =======================    renormalization  ======================#
        new_maxes = torch.max(cur_maxes, prev_maxes) # update m(x)
        # renormalization factor for the previous block
        renorm_prev = torch.exp(prev_maxes - new_maxes)
        # renormalization factor for the current block
        renorm_cur = torch.exp(cur_maxes - new_maxes)

        # update normalization factor l(x)
        new_sums = renorm_prev * prev_sums + renorm_cur * cur_sums

        o = (prev_out * prev_sums * renorm_prev +
            renorm_cur * exp_values) / new_sums

        return new_maxes, new_sums, o

    def forward(self):
         N = self.n // self.ktn

         print(N)

         prev_maxes = torch.zeros(self.m, 1, device='cpu')
         prev_sums = torch.zeros(self.m, 1, device='cpu')

         o = self.O.view(self.m, self.p)

         for n in range(N):
            q = self.Q.view(self.m, self.k) # m * k

            k = self.K[n * self.k * self.ktn: (n + 1) * self.k * self.ktn].view(self.k, self.ktn)
            v = self.V[n * self.p * self.ktn: (n + 1) * self.p * self.ktn].view(self.ktn, self.p)
            
            attn_weights = torch.mm(q, k) # m * ktn

            # print('attn_weights:', attn_weights)

            # reduce maxes
            cur_maxes, _ = torch.max(attn_weights, dim=-1, keepdim=True)
            exp_weights = torch.exp(attn_weights - cur_maxes)
            # unnormalized attention score @ values
            exp_values = exp_weights @ v
            # move the normalization step to the very end of the attention computation.
            cur_sums = torch.sum(exp_weights, dim=-1, keepdim=True) # l(x_cur)

            # =======================    renormalization  ======================#
            new_maxes = torch.max(cur_maxes, prev_maxes) # update m(x)
            # print('new_maxes: ', new_maxes)
            # renormalization factor for the previous block
            renorm_prev = torch.exp(prev_maxes - new_maxes)
            # renormalization factor for the current block
            renorm_cur = torch.exp(cur_maxes - new_maxes)

            # update normalization factor l(x)
            new_sums = renorm_prev * prev_sums + renorm_cur * cur_sums
            # print('new_sums: ', new_sums)
            # print('cur_sums: ', cur_sums)

            o = (o * prev_sums * renorm_prev +
                renorm_cur * exp_values) / new_sums

         self.O = o

         return self.O


class TestFlashAttention(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1234)

    def test_flash_attention(self):
        m = 64
        n = 64
        k = 128 
        p = 128

        ktm = 64
        ktn = 64
        ktk = 128
        ktp = 128

        Q = torch.randn(m, k, device='cpu')
        K = torch.randn(k, n, device='cpu')
        V = torch.randn(n, p, device='cpu')
        O = torch.empty(m, p, device='cpu')

        dQ = Q.to('cuda')
        dK = K.to('cuda')
        dV = V.to('cuda')
        dO = O.to('cuda')

        print(K.half().shape, K.half().t().shape)

        flash_attn = FlashAttention(Q.half().flatten(), K.half().flatten(), V.half().flatten(), O.half().flatten(), m, n, k, p, ktm, ktn, ktk, ktp)

        ref_o = flash_attn.forward()

        print(ref_o)


        dQ = dQ.half().flatten()
        dK = dK.half().t().flatten()
        dV = dV.half().t().flatten()
        dO = dO.half().flatten()

        flash_attention_fwd(dQ, dK, dV, dO, m, n, k, p)

        print(dO.view(m, p))

if __name__ == "__main__":

    unittest.main()

    