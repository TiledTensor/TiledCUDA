#pragma once

#include "cuda_utils.hpp"

#include <torch/script.h>

namespace tiledcuda::kernels {
template <typename Element, typename KeTraits>
__global__ void dyn_cute_batched_gemm_kernel(const Element* dA,
                                             const Element* dB, Element* dC,
                                             int m, int n, int k);

template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape>
void cute_batched_gemm(const Element* a, const Element* b, Element* c, int m,
                       int n, int k, int batch_count);

void custom_batched_gemm_op(const torch::Tensor& a, const torch::Tensor& b,
                            torch::Tensor& c, int64_t m, int64_t n, int64_t k,
                            int64_t batch_count);
}  // namespace tiledcuda::kernels
