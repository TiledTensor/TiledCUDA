import torch
import torch.nn as nn
from typing import Tuple
import sys
sys.path.append('./')


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
        i = torch.sigmoid(torch.mm(self.wi[0], self.xi) +
                          torch.mm(self.ui[0], self.h0))
        # Forget gate
        f = torch.sigmoid(torch.mm(self.wi[1], self.xi) +
                          torch.mm(self.ui[1], self.h0))
        # Output gate
        o = torch.sigmoid(torch.mm(self.wi[2], self.xi) +
                          torch.mm(self.ui[2], self.h0))
        # Cell gate
        c = torch.tanh(torch.mm(self.wi[3], self.xi) +
                       torch.mm(self.ui[3], self.h0))

        self.c1 = f * self.c0 + i * c
        self.h1 = o * torch.tanh(self.c1)

        return self.c1, self.h1


if __name__ == "__main__":
    import pytiledcuda

    hidden = 256
    batch = 256

    w = torch.randn(4, hidden, hidden, device='cuda')
    x = torch.randn(hidden, batch, device='cuda')
    u = torch.randn(4, hidden, hidden, device='cuda')
    c0 = torch.randn(hidden, batch, device='cuda')
    h0 = torch.randn(hidden, batch, device='cuda')
    c1 = torch.empty(hidden, batch, device='cuda')
    h1 = torch.empty(hidden, batch, device='cuda')

    w_data = w.flatten().half()
    x_data = x.T.flatten().half()
    u_data = u.flatten().half()
    c0_data = c0.flatten().half()
    h0_data = h0.T.flatten().half()
    c1_data = c1.flatten().half()
    h1_data = h1.flatten().half()

    pytiledcuda.lstm_cell(w_data, x_data, u_data, c0_data,
                          h0_data, c1_data, h1_data, hidden, batch)

    ref_lstm = FineGrainedOpLstmCell(
        w.half(), x.half(), u.half(), c0.half(), h0.half(), c1.half(), h1.half(), batch, hidden)
    ref_c, ref_h = ref_lstm.forward()

    ref_c = ref_c.flatten()
    ref_h = ref_h.flatten()

    assert torch.allclose(c1_data, ref_c, atol=1e-1)
    assert torch.allclose(h1_data, ref_h, atol=1e-1)

    print("Lstm Cell test passed.")
