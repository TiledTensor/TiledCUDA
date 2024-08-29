import unittest

import torch

import context
from pytiledcuda import flash_attention_fwd

class FlashAttention:

    def __init__(self, Q, K, V, O, m, n, k, p, kTM, kTN, kTK, kTP):
        self.m = m
        self.n = n
        self.k = k
        self.p = p

        self.kTM = kTM
        self.kTN = kTN
        self.kTK = kTK
        self.kTP = kTP

        self.Q = Q # m * k
        self.K = K # n * k
        self.V = V # n * p
        self.O = O # m * p

    
    # def attn_func(self, prev_maxes, prev_sums, prev_out, q, k, v):
    #     attn_weights = q @ k.T

    #     # reduce maxes
    #     cur_maxes = torch.max(attn_weights, dim=-1, keepdim=True)
    #     exp_weights = torch.exp(attn_weights - cur_maxes)
    #     # unnormalized attention score @ values
    #     exp_values = exp_weights @ v
    #     # move the normalization step to the very end of the attention computation.
    #     cur_sums = torch.sum(exp_weights, dim=-1, keepdim=True) # l(x_cur)

    #     # =======================    renormalization  ======================#
    #     new_maxes = torch.max(cur_maxes, prev_maxes) # update m(x)
    #     # renormalization factor for the previous block
    #     renorm_prev = torch.exp(prev_maxes - new_maxes)
    #     # renormalization factor for the current block
    #     renorm_cur = torch.exp(cur_maxes - new_maxes)

    #     # update normalization factor l(x)
    #     new_sums = renorm_prev * prev_sums + renorm_cur * cur_sums

    #     o = (prev_out * prev_sums * renorm_prev +
    #         renorm_cur * exp_values) / new_sums

    #     return new_maxes, new_sums, o

    def forward(self):
         N = self.n // self.kTN

         prev_maxes = torch.zeros(self.m, 1, device='cpu')
         prev_sums = torch.zeros(self.m, 1, device='cpu')

         o = self.O.view(self.m, self.p)

         dK = self.K.view(self.k, self.n)
         dV = self.V.view(self.n, self.p)

         ks = torch.chunk(dK, N, dim=-1)
         vs = torch.chunk(dV, N, dim=-2)


         for n in range(N):
            q = self.Q.view(self.m, self.k) # m * k

            k = ks[n]
            v = vs[n]

            # print(k.shape)
            
            attn_weights = torch.mm(q, k) # m * ktn

            # print('attn_weights:', attn_weights)

            # reduce maxes
            cur_maxes, _ = torch.max(attn_weights, dim=-1, keepdim=True)
            exp_weights = torch.exp(attn_weights - cur_maxes)
            # unnormalized attention score @ values
            exp_values = torch.mm(exp_weights, v)
            # move the normalization step to the very end of the attention computation.
            cur_sums = torch.sum(exp_weights, dim=-1, keepdim=True) # l(x_cur)

            # =======================    renormalization  ======================#
            new_maxes = torch.max(cur_maxes, prev_maxes) # update m(x)
            # print('new_maxes: ', new_maxes.flatten())
            # renormalization factor for the previous block
            renorm_prev = torch.exp(prev_maxes - new_maxes)
            # renormalization factor for the current block
            renorm_cur = torch.exp(cur_maxes - new_maxes)

            # update normalization factor l(x)
            new_sums = renorm_prev * prev_sums + renorm_cur * cur_sums

            # print('prev_sums: ', prev_sums.flatten())
            # print('cur_sums: ', cur_sums.flatten())
            # print('new_sums: ', new_sums.flatten())

            # print('prev_maxes: ', prev_maxes.flatten())
            # print('cur_maxes: ', cur_maxes.flatten())
            # print('renorm_prev: ', renorm_prev.flatten())
            # print('renorm_cur: ', renorm_cur.flatten())

            # print('exp_values: ', exp_values.flatten())

            # print('exp_weights: ', exp_weights.flatten())
            # print('v: ', v.flatten())

            lhs_o = o * prev_sums * renorm_prev
            rhs_o = renorm_cur * exp_values

            print('lhs_o: ', lhs_o.flatten())
            print('rhs_o: ', rhs_o.flatten())

            o = (o * prev_sums * renorm_prev +
                renorm_cur * exp_values) / new_sums

            prev_sums = new_sums
            prev_maxes = new_maxes

         self.O = o

         return self.O




class TestFlashAttention(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1234)
    
    def run_flash_attention(self, m, n, k, p, kTM, kTN, kTK, kTP):

        Q = torch.randn(m, k, device='cpu')
        K = torch.randn(k, n, device='cpu')
        V = torch.randn(n, p, device='cpu')
        O = torch.empty(m, p, device='cpu')

        flash_attn = FlashAttention(Q.half().flatten(), K.half().flatten(), V.half().flatten(), O.half().flatten(), m, n, k, p, kTM, kTN, kTK, kTP)

        ref_o = flash_attn.forward().half()

        dQ = Q.to('cuda')
        dK = K.to('cuda')
        dV = V.to('cuda')
        dO = O.to('cuda')

        dQ = dQ.half().flatten()
        dK = dK.half().t().flatten()
        dV = dV.half().t().flatten()
        dO = dO.half().flatten()

        flash_attention_fwd(dQ, dK, dV, dO, m, n, k, p)

        print('ref_o: ', ref_o)
        print('dO: ', dO.view(m, p))

        hO = dO.view(m, p).cpu()

    
        # Compare elements one by one and print the different numbers.
        # for i in range(m):
        #     for j in range(p):
        #         if abs(hO[i][j] - ref_o[i][j]) > 8e-2:
        #             print('(', i, ', ', j, ')')
        #             print('hO: ', hO[i][j])
        #             print('ref_o: ', ref_o[i][j])


    # def test_flash_attention_v0(self):
    #     m = 64
    #     n = 64
    #     k = 128 
    #     p = 128

    #     kTM = 64
    #     kTN = 64
    #     kTK = 128
    #     kTP = 128

    #     self.run_flash_attention(m, n, k, p, kTM, kTN, kTK, kTP)

    def test_flash_attention_v1(self):
        m = 64
        n = 128
        k = 128 
        p = 128

        kTM = 64
        kTN = 64
        kTK = 128
        kTP = 128

        self.run_flash_attention(m, n, k, p, kTM, kTN, kTK, kTP)

if __name__ == "__main__":

    unittest.main()

    