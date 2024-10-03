from typing import Tuple
import unittest

import torch
import torch.nn as nn

import context

from pytiledcuda import lstm_cell


class FineGrainedOpLstmCell(nn.Module):

    def __init__(self, w, x, u, c0, h0, c1, h1, batch, hidden):
        super(FineGrainedOpLstmCell, self).__init__()
        self.wi = w
        self.xi = x
        self.ui = u
        self.c0 = c0
        self.h0 = h0
        self.c1 = c1
        self.h1 = h1
        self.batch = batch
        self.hidden = hidden

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Implement `lstm_cell` in Python.
        # Input gate
        i = torch.sigmoid(
            torch.mm(self.wi[0], self.xi) + torch.mm(self.ui[0], self.h0))
        # Forget gate
        f = torch.sigmoid(
            torch.mm(self.wi[1], self.xi) + torch.mm(self.ui[1], self.h0))
        # Output gate
        o = torch.sigmoid(
            torch.mm(self.wi[2], self.xi) + torch.mm(self.ui[2], self.h0))
        # Cell gate
        c = torch.tanh(
            torch.mm(self.wi[3], self.xi) + torch.mm(self.ui[3], self.h0))

        self.c1 = f * self.c0 + i * c
        self.h1 = o * torch.tanh(self.c1)

        return self.c1, self.h1


class TestLstmCell(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1234)

    def test_lstm_cell(self):
        device = torch.device("cuda")
        dtype = torch.float16

        hidden = 256
        batch = 256

        w = torch.randn(4, hidden, hidden, device=device, dtype=dtype)
        x = torch.randn(hidden, batch, device=device, dtype=dtype)
        u = torch.randn(4, hidden, hidden, device=device, dtype=dtype)
        c0 = torch.randn(hidden, batch, device=device, dtype=dtype)
        h0 = torch.randn(hidden, batch, device=device, dtype=dtype)
        c1 = torch.empty(hidden, batch, device=device, dtype=dtype)
        h1 = torch.empty(hidden, batch, device=device, dtype=dtype)

        w_data = w.flatten()
        x_data = x.T.flatten()
        u_data = u.flatten()
        c0_data = c0.flatten()
        h0_data = h0.T.flatten()
        c1_data = c1.flatten()
        h1_data = h1.flatten()

        lstm_cell(w_data, x_data, u_data, c0_data, h0_data, c1_data, h1_data,
                  hidden, batch)

        ref_lstm = FineGrainedOpLstmCell(w, x, u, c0, h0, c1, h1, batch,
                                         hidden)
        ref_c, ref_h = ref_lstm.forward()

        ref_c = ref_c.flatten()
        ref_h = ref_h.flatten()

        assert torch.allclose(c1_data, ref_c, atol=1e-1)
        assert torch.allclose(h1_data, ref_h, atol=1e-1)

        print("Lstm Cell test passed.")


if __name__ == "__main__":
    unittest.main()
