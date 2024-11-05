#pragma once
#include "cuda_utils.hpp"

#include <torch/script.h>

namespace tiledcuda::kernels {


template <typename InType, typename AccType,                  //
          const int kM, const int kN, const int kK,           //
          const int kTM, const int kTN, const int kTK,        //
          typename GIteratorA, typename SIteratorA,           //
          typename SharedA, typename RegA,                    //
          typename G2SLoaderA, typename S2RLoaderA,           //
          typename GIteratorB, typename SIteratorB,           //
          typename SharedB, typename RegB,                    //
          typename G2SLoaderB, typename S2RLoaderB,           //
          typename GlobalC, typename SharedC, typename RegC,  //
          typename R2SStorerC, typename S2GStorerC>
__global__ void gemm(const InType* dA, const InType* dB, AccType* dC);

template <typename InType, typename AccType, typename WholeShape,
          typename CtaTileShape>
void run_gemm(const InType* dA, const InType* dB, AccType* dC);

void custom_gemm_op(const torch::Tensor& A, const torch::Tensor& B,
                    torch::Tensor& C, int64_t m, int64_t n, int64_t k);

}  // namespace tiledcuda::kernels
