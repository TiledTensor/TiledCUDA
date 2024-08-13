#include "util.hpp"

template <typename InType, typename AccType, typename IteratorA, typename RegA,
          typename LoaderA, typename IteratorB, typename RegB, typename LoaderB,
          typename IteratorC, typename RegC, typename LoaderC, typename RegAcc,
          typename RegAccCast, typename GlobalD, typename RegD,
          typename DStorer, typename ConvertAcc>
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
    RegAccCast acc_half;

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
        cast_acc(acc, acc_half);

        // Compute GEMM using acc_half, C tiles
        compute::gemm_(acc_half, rC, rD);
    }

    __syncthreads();

    // Store D tile from register to global.
    GlobalD gD(dD);
    DStorer storer_d;
    storer_d(rD, gD);
}

int main() {
    using InType = __half;
    using AccType = float;

    const int kM = 128;
    const int kN = 128;
    const int kK = 128;
    const int kP = 128;

    // initialize data
    thrust::host_vector<InType> h_a(kM * kK);
    for (int i = 0; i < h_a.size(); ++i) {
        h_a[i] = static_cast<InType>(rand_float());
    }

    thrust::host_vector<InType> h_b(kK * kN);
    for (int i = 0; i < h_b.size(); ++i) {
        h_b[i] = static_cast<InType>(rand_float());
    }

    thrust::host_vector<InType> h_acc(kM * kN);
    thrust::fill(h_acc.begin(), h_acc.end(), 0.);

    thrust::host_vector<InType> h_c(kN * kP);
    for (int i = 0; i < h_c.size(); ++i) {
        h_c[i] = static_cast<InType>(rand_float());
    }

    thrust::host_vector<AccType> h_d(kM * kP);
    thrust::fill(h_c.begin(), h_c.end(), 0.);

    thrust::device_vector<InType> d_a = h_a;
    thrust::device_vector<InType> d_b = h_b;
    thrust::device_vector<InType> d_c = h_c;
    thrust::device_vector<AccType> d_d = h_d;

    const InType* A = thrust::raw_pointer_cast(d_a.data());
    const InType* B = thrust::raw_pointer_cast(d_b.data());
    const InType* C = thrust::raw_pointer_cast(d_c.data());
    AccType* D = thrust::raw_pointer_cast(d_d.data());

    using Config = B2BGemmTraits<InType, AccType, B2BGemmShape<kM, kN, kK, kP>>;

    using RegA = typename Config::RegA;
    using RegB = typename Config::RegB;
    using RegC = typename Config::RegC;
    using RegD = typename Config::RegD;
    using RegAcc = typename Config::RegAcc;
    using RegAccCast = typename Config::RegAccCast;

    using IteratorA = typename Config::IteratorA;
    using IteratorB = typename Config::IteratorB;
    using IteratorC = typename Config::IteratorC;

    using LoaderA = typename Config::ALoader;
    using LoaderB = typename Config::BLoader;
    using LoaderC = typename Config::CLoader;
    using DStorer = typename Config::DStorer;

    using ConvertAcc = typename Config::ConvertHalf;

    dim3 grid(1, 1, 1);
    dim3 block(Config::kThreads, 1, 1);

    simple_back2back_gemm<InType, AccType, IteratorA, RegA, LoaderA, IteratorB,
                          RegB, LoaderB, IteratorC, RegC, LoaderC, RegAcc,
                          RegAccCast, typename Config::GlobalD, RegD, DStorer,
                          ConvertAcc><<<grid, block>>>(A, B, C, D);

    cudaDeviceSynchronize();

    h_d = d_d;

    {  // check correctness
        thrust::host_vector<AccType> h_d2(kM * kP);
        thrust::fill(h_d2.begin(), h_d2.end(), 0.);
        naive_back2back_gemm(kM, kN, kK, kP,
                             thrust::raw_pointer_cast(h_a.data()),
                             thrust::raw_pointer_cast(h_b.data()),
                             thrust::raw_pointer_cast(h_c.data()), h_d2.data(),
                             thrust::raw_pointer_cast(h_acc.data()));

        bool passed =
            check_results(thrust::raw_pointer_cast(h_d.data()),
                          thrust::raw_pointer_cast(h_d2.data()), kM * kP);

        if (passed) {
            std::cout << "Test passed." << std::endl;
        } else {
            std::cerr << "Test failed." << std::endl;
        }
    }

    return 0;
}