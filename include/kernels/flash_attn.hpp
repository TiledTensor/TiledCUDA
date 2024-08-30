#pragma once

#include "cuda_utils.hpp"

#include <torch/script.h>

namespace tiledcuda::kernels {

template <typename InType,
          typename AccType,                                      //
          typename OutType,                                      //
          typename GIteratorQ, typename SharedQ, typename RegQ,  //
          typename SharedQLoader, typename RegQLoader,           //
          typename GIteratorK, typename SharedK, typename RegK,  //
          typename SharedKLoader, typename RegKLoader,           //
          typename GIteratorV, typename SharedV, typename RegV,  //
          typename SharedVLoader, typename RegVLoader,           //
          typename RegAcc, typename RegAccCast, typename GlobalO, typename RegO,
          typename RegOCast, typename OStorer, typename ConvertAcc,
          typename ConvertO, typename RegVec, typename CopyVec, typename RowMax,
          typename RowSum, typename BroadcastSub, typename BroadcastMul,
          typename BroadcastDiv, typename BlockExp, typename BlockAdd,
          typename VecMax, typename VecAdd, typename VecSub, typename VecMul,
          typename VecExp>
__global__ void flash_attention(const InType* dQ, const InType* dK,
                                const InType* dV, InType* dO, int kM, int kN,
                                int kK, int kP, int kTM, int kTN, int kTK,
                                int kTP);

template <typename InType, typename AccType, typename OutType,
          typename WholeShape, typename CtaTileShape, const int kBatch>
void run_flash_attention(const InType* dQ, const InType* dK, const InType* dV,
                         OutType* dO);

void custom_flash_attention_op(const torch::Tensor& Q, const torch::Tensor& K,
                               const torch::Tensor& V, torch::Tensor& O,
                               int64_t m, int64_t n, int64_t k, int64_t p);

}  // namespace tiledcuda::kernels
