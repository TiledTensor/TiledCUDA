#pragma once
#include "cuda_utils.hpp"

namespace tiledcuda::kernels {

template <typename Element, typename KeTraits>
__global__ void dyn_back2back_gemm(const Element* d_a, const Element* d_b,
                                   const Element* d_c, Element* d_d, int m,
                                   int n, int k, int p);

}