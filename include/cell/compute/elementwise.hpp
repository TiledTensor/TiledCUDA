#pragma once

#include "cuda_utils.hpp"

#include <cute/config.hpp>
#include <cute/tensor.hpp>

namespace tiledcuda::cell::compute {

template <class XEngine, class XLayout>
__forceinline__ __device__ void cute_tanh(
    cute::Tensor<XEngine, XLayout>& tensor) {
#pragma unroll
    for (int i = 0; i < size(tensor); ++i) {
        tensor(i) = tanh(tensor(i));
    }
}

template <class XEngine, class XLayout>
__forceinline__ __device__ void cute_sigmod(
    cute::Tensor<XEngine, XLayout>& tensor) {
#pragma unroll
    for (int i = 0; i < size(tensor); ++i) {
        tensor(i) = 1.0 / (1.0 + exp(-tensor(i)));
    }
}
}  // namespace tiledcuda::cell::compute