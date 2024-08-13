#include "util.hpp"

template <typename InType, typename AccType, typename IteratorA, typename RegA,
          typename LoaderA, typename IteratorB, typename RegB, typename LoaderB,
          typename IteratorC, typename RegC, typename LoaderC, typename RegAcc,
          typename GlobalD, typename RegD, typename DStorer,
          typename ConvertAcc>
__global__ void simple_back2back_gemm(const InType* dA, const InType* dB,
                                      const InType* dC, AccType* dD) {
    IteratorA iter_g2r_a(dA);
    RegA rA;
    LoaderA loader_a;

    IteratorB iter_g2r_b(dB);
    RegB rB;
    LoaderB loader_b;

    IteratorC iter_g2r_c(dC);
    RegC rC;
    LoaderC loader_c;

    RegAcc acc;

    RegD rD;

    for (int n = 0; n < IteratorC::sc0; ++n) {
        // iterate over n, C: [N, P]
        for (int k = 0; k < IteratorA::sc1; ++k) {
            // iterate over k, A: [M, K]
            // Load A, B tiles from global to register
            loader_a(iter_g2r_a(k), rA);
            loader_b(iter_g2r_b(k), rB);
            __syncthreads();

            // Compute GEMM using A, B tiles
            compute::gemm_(rA, rB, acc);
        }
        // Load C tile from global to register
        loader_c(iter_g2r_c(n), rC);
        __syncthreads();

        // Convert acc to half precision
        ConvertAcc cast_acc;
        auto acc_half = cast_acc(acc);

        // Compute GEMM using acc_half, C tiles
        compute::gemm_(acc_half, rC, rD);
    }

    __syncthreads();

    // Store D tile from register to global.
    GlobalD gD(dD);
    DStorer storer_d;
    storer_d(rD, gD);
}