import torch

from compile import Compile

__all__ = [
    "gemm_func",
]


class GemmFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, B, C, M, N, K, kM, kN):
        builder = Compile(file_name="gemm.cu", tmp_dir="tmp")
        lib_name = builder.compile(M, N, K, kM, kN)

        if lib_name is None:
            raise RuntimeError("Failed to compile the library.")

        builder.apply(lib_name, [A, B, C], device=0)
        return C


gemm_func = GemmFunc.apply
