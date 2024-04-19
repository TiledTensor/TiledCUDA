import torch

torch.ops.load_library("build/libtiledcuda.so")


def gemm(a, b, c, m, n, k):
    torch.ops.tiledcuda.gemm(a, b, c, m, n, k)


def scatter_nd(scatter_data, scatter_indices, scatter_updates):
    torch.ops.tiledcuda.scatter_nd(
        scatter_data, scatter_updates, scatter_indices)
