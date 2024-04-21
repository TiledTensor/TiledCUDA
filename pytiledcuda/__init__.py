import torch

torch.ops.load_library("build/libtiledcuda.so")


def scatter_nd(scatter_data, scatter_indices, scatter_updates):
    torch.ops.tiledcuda.scatter_nd(
        scatter_data, scatter_updates, scatter_indices)


def gemm(a, b, c, m, n, k):
    torch.ops.tiledcuda.gemm(a, b, c, m, n, k)


def batched_gemm(a, b, c, m, n, k, batch_count):
    torch.ops.tiledcuda.batched_gemm(a, b, c, m, n, k, batch_count)


def back2back_gemm(a, b, c, d, m, n, k, p):
    torch.ops.tiledcuda.back2back_gemm(a, b, c, d, m, n, k, p)
