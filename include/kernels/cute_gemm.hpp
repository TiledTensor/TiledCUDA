#pragma once
#include "cuda_utils.hpp"

#include <torch/script.h>

namespace tiledcuda::kernels {

template <typename Element, typename KeTraits>
__global__ void dyn_cute_gemm_kernel(const Element* dA, const Element* dB,
                                     Element* dC, int m, int n, int k);

template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape>
void cute_gemm(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c,
               int m, int n, int k);

void custom_gemm_op(const torch::Tensor& a, const torch::Tensor& b,
                    torch::Tensor& c, int64_t m, int64_t n, int64_t k);

}  // namespace tiledcuda::kernels