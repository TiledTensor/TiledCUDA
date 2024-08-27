import torch

torch.ops.load_library("build/src/libtiledcuda.so")


def scatter_nd(scatter_data, scatter_indices, scatter_updates):
    torch.ops.tiledcuda.scatter_nd(scatter_data, scatter_updates,
                                   scatter_indices)


def gemm(a, b, c, m, n, k):
    torch.ops.tiledcuda.gemm(a, b, c, m, n, k)


def batched_gemm(a, b, c, m, n, k, batch_count):
    torch.ops.tiledcuda.batched_gemm(a, b, c, m, n, k, batch_count)


def back2back_gemm(a, b, c, d, m, n, k, p):
    torch.ops.tiledcuda.back2back_gemm(a, b, c, d, m, n, k, p)


def lstm_cell(w, x, u, c0, h0, c1, h1, batch, hidden):
    torch.ops.tiledcuda.lstm_cell(w, x, u, c0, h0, c1, h1, batch, hidden)

def flash_attention_fwd(Q, K, V, O, m, n, k, p):
    torch.ops.tiledcuda.flash_attention_fwd(Q, K, V, O, m, n, k, p)
