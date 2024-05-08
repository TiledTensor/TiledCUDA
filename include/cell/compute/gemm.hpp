#pragma once
#include "cuda_utils.hpp"

#include <cute/tensor.hpp>

namespace tiledcuda::cell::compute {
template <typename TensorA, typename TensorB, typename TensorAcc,
          typename TiledMma>
__forceinline__ __device__ void gemm(const TiledMma& mma, const TensorA& a,
                                     const TensorB& b, TensorAcc& acc) {
    cute::gemm(mma, a, b, acc);  // compute
}
}  // namespace tiledcuda::cell::compute
