#pragma once
#include "cuda_utils.hpp"

#include <torch/script.h>

namespace tiledcuda::kernels {

template <typename Element, typename KeTraits>
__global__ void dyn_back2back_gemm(const Element* d_a, const Element* d_b,
                                   const Element* d_c, Element* d_d, int m,
                                   int n, int k, int p);

template <typename Element, typename CtaTileShape, typename WarpShape>
void cute_back2back_gemm(const Element* d_a, const Element* d_b,
                         const Element* d_c, Element* d_d, int m, int n, int k,
                         int p);

void custom_back2back_op(const torch::Tensor& a, const torch::Tensor& b,
                         const torch::Tensor& c, torch::Tensor& d, int64_t m,
                         int64_t n, int64_t k, int64_t p);

}  // namespace tiledcuda::kernels
