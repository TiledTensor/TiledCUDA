#pragma once
#include "cuda_utils.hpp"

#include <cute/tensor.hpp>

namespace tiledcuda::cell {
template <typename TensorA, typename TensorB, typename TensorAcc>
__forceinline__ __device__ void gemm(const auto& mma, const TensorA& a,
                                     const TensorB& b, TensorAcc& acc) {
    cute::gemm(mma, a, b, acc);  // compute
}

}  // namespace tiledcuda::cell