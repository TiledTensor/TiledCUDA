#pragma once

using namespace tiledcuda;
using namespace tiledcuda::cell;
using namespace tiledcuda::cell::compute;

template <typename InType, typename AccType,                    //
          const int kM, const int kN, const int kK,             //
          const int kTM, const int kTN,                         //
          typename IteratorA, typename RegA, typename ALoader,  //
          typename IteratorB, typename RegB, typename BLoader,  //
          typename GlobalC, typename RegC, typename CStorer>
__global__ void gemm(const InType* dA, const InType* dB, AccType* dC) {
    int offset_a = blockIdx.x * kTM * kK;
    int offset_b = blockIdx.y * kTN * kK;
    int offset_c = blockIdx.x * kTM * kN + blockIdx.y * kTN;

    IteratorA gAs(dA + offset_a);
    RegA rA;
    ALoader loader_a;

    IteratorB gBs(dB + offset_b);
    RegB rB;
    BLoader loader_b;

    RegC acc;
    GlobalC gC(dC + offset_c);
    CStorer storer_c;

    for (int k = 0; k < IteratorA::sc1; ++k) {
        loader_a(gAs(k), rA);
        loader_b(gBs(k), rB);
        __syncthreads();

        compute::gemm_(rA, rB, acc);
    }
    __syncthreads();

    storer_c(acc, gC);
}
