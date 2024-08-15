#include "util.hpp"

template <typename InType, typename AccType,                     //
          typename GIteratorA, typename SharedA, typename RegA,  //
          typename SharedALoader, typename RegALoader,           //
          typename GIteratorB, typename SharedB, typename RegB,  //
          typename SharedBLoader, typename RegBLoader,           //
          typename GIteratorC, typename SharedC, typename RegC,  //
          typename SharedCLoader, typename RegCLoader,           //
          typename RegAcc, typename RegAccCast, typename GlobalD, typename RegD,
          typename DStorer, typename ConvertAcc>
__global__ void KeBack2BackGemm(const InType* dA, const InType* dB,
                                const InType* dC, AccType* dD, int kM, int kN,
                                int kK, int kP, int kTM, int kTN, int kTK,
                                int kTP) {
    // Advance to the global data tile to the current CTA.
    const InType* A = dA + blockIdx.x * (kTM * kK);
    const InType* B = dB;
    const InType* gC_ptr = dC + blockIdx.y * (kTP * kN);
    AccType* gD_ptr = dD + blockIdx.x * (kTM * kP) + (blockIdx.y * kTP);

    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<InType*>(shared_buf);

    InType* sA_ptr = shm;
    InType* sB_ptr = shm + SharedA::kNumel;
    InType* sC_ptr = shm + SharedA::kNumel + SharedB::kNumel;

    GIteratorA gAs(A);
    SharedA sA(sA_ptr);
    RegA rA;

    SharedALoader load_sa;
    RegALoader load_ra;

    GIteratorB gBs(B);
    SharedB sB(sB_ptr);
    RegB rB;

    SharedBLoader load_sb;
    RegBLoader load_rb;

    GIteratorC gCs(gC_ptr);
    SharedC sC(sC_ptr);

    SharedCLoader load_sc;
    RegCLoader load_rc;
    RegC rC;

    RegD rD;
    RegAcc acc;
    RegAccCast acc_half;

    for (int n = 0; n < GIteratorC::sc0; ++n) {
        load_sc(gCs(n), sC);

        for (int k = 0; k < GIteratorA::sc1; ++k) {
            load_sa(gAs(k), sA);
            load_sb(gBs(k, n), sB);
            __copy_async();
            __syncthreads();

            load_ra(sA, rA);
            load_rb(sB, rB);
            __syncthreads();

            compute::gemm_(rA, rB, acc);
        }
        load_rc(sC, rC);
        __syncthreads();

        ConvertAcc cast_acc;  // Convert acc to half precision
        cast_acc(acc, acc_half);

        compute::gemm_(acc_half, rC, rD);
    }
    __syncthreads();

    GlobalD gD(gD_ptr);
    DStorer storer_d;  // Store D tile from register to global.
    storer_d(rD, gD);
}

template <typename WholeShape, typename CtaTileShape>
void run() {
    using InType = __half;
    using AccType = float;

    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;
    static constexpr int kP = dim_size<3, WholeShape>;

    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;
    static constexpr int kTP = dim_size<3, CtaTileShape>;

    static_assert(kK == kTK,
                  "The current implementation requires kTK == K for now.");
    static_assert(kP == kTP,
                  "The current implementation requires kTP == P for now.");

    // initialize data
    thrust::host_vector<InType> h_a(kM * kK);

    for (int i = 0; i < h_a.size(); ++i)
        h_a[i] = static_cast<InType>(rand_float());

    thrust::host_vector<InType> h_b(kK * kN);
    for (int i = 0; i < h_b.size(); ++i)
        h_b[i] = static_cast<InType>(rand_float());

    thrust::host_vector<InType> h_c(kN * kP);
    for (int i = 0; i < h_c.size(); ++i)
        h_c[i] = static_cast<InType>(rand_float());

    thrust::host_vector<AccType> h_d(kM * kP);
    thrust::fill(h_d.begin(), h_d.end(), 0.);

    thrust::device_vector<InType> d_a = h_a;
    thrust::device_vector<InType> d_b = h_b;
    thrust::device_vector<InType> d_c = h_c;
    thrust::device_vector<AccType> d_d = h_d;

    const InType* A = thrust::raw_pointer_cast(d_a.data());
    const InType* B = thrust::raw_pointer_cast(d_b.data());
    const InType* C = thrust::raw_pointer_cast(d_c.data());
    AccType* D = thrust::raw_pointer_cast(d_d.data());

    using Config = B2BGemmTraits<InType, AccType, WholeShape, CtaTileShape>;

    using RegA = typename Config::RegA;
    using RegB = typename Config::RegB;
    using RegC = typename Config::RegC;
    using RegD = typename Config::RegD;
    using RegAcc = typename Config::RegAcc;
    using RegAccCast = typename Config::RegAccCast;

    using GIteratorA = typename Config::GIteratorA;
    using SharedA = typename Config::SharedA;
    using SharedALoader = typename Config::SharedALoader;
    using RegALoader = typename Config::RegALoader;

    using GIteratorB = typename Config::GIteratorB;
    using SharedB = typename Config::SharedB;
    using SharedBLoader = typename Config::SharedBLoader;
    using RegBLoader = typename Config::RegBLoader;

    using GIteratorC = typename Config::GIteratorC;
    using SharedC = typename Config::SharedC;
    using SharedCLoader = typename Config::SharedCLoader;
    using RegCLoader = typename Config::RegCLoader;

    using DStorer = typename Config::DStorer;

    using ConvertAcc = typename Config::ConvertHalf;

    int block_x = CeilDiv<kM, kTM>;
    int block_y = CeilDiv<kP, kTP>;

    dim3 grid(block_x, block_y, 1);
    dim3 block(Config::kThreads, 1, 1);

    int shm_input = (kTM * kTK + kTK * kTN + kTN * kTP);
    int shm_output = kTM * kTP;
    int shm_size = shm_input < shm_output ? shm_output * sizeof(InType)
                                          : shm_input * sizeof(InType);

    auto kernel = &KeBack2BackGemm<InType, AccType,            //
                                   GIteratorA, SharedA, RegA,  //
                                   SharedALoader, RegALoader,  //
                                   GIteratorB, SharedB, RegB,  //
                                   SharedBLoader, RegBLoader,  //
                                   GIteratorC, SharedC, RegC,  //
                                   SharedCLoader, RegCLoader,  //
                                   RegAcc, RegAccCast, typename Config::GlobalD,
                                   RegD, DStorer, ConvertAcc>;

    if (shm_size > 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    }

    kernel<<<grid, block, shm_size, 0>>>(A, B, C, D, kM, kN, kK, kP, kTM, kTN,
                                         kTK, kTP);
    h_d = d_d;
    cudaDeviceSynchronize();

    thrust::host_vector<InType> h_acc(kM * kN);
    thrust::fill(h_acc.begin(), h_acc.end(), 0.);

    thrust::host_vector<AccType> h_d2(kM * kP);
    thrust::fill(h_d2.begin(), h_d2.end(), 0.);
    naive_back2back_gemm(kM, kN, kK, kP, thrust::raw_pointer_cast(h_a.data()),
                         thrust::raw_pointer_cast(h_b.data()),
                         thrust::raw_pointer_cast(h_c.data()), h_d2.data(),
                         thrust::raw_pointer_cast(h_acc.data()));
    cudaDeviceSynchronize();

    float* data = thrust::raw_pointer_cast(h_d.data());
    float* ground_truth = thrust::raw_pointer_cast(h_d2.data());

#ifdef DEBUG
    std::cout << "GIteratorA: " << GIteratorA{} << std::endl
              << "GIteratorB: " << GIteratorB{} << std::endl
              << "GIteratorC: " << GIteratorC{} << std::endl;

    std::cout << "block_x: " << block_x << std::endl
              << "block_y: " << block_y << std::endl;

    printf("ours:\n");
    for (int i = 0; i < h_d.size(); ++i) {
        printf("%.2f, ", data[i]);
        if (i && (i + 1) % 16 == 0) printf("\n");
    }
    printf("\nground_truth:\n");
    for (int i = 0; i < h_d.size(); ++i) {
        printf("%.2f, ", ground_truth[i]);
        if (i && (i + 1) % 16 == 0) printf("\n");
    }
    std::cout << "Done" << std::endl;
#endif

    if (check_results(data, ground_truth, kM * kP)) {
        std::cout << "Test passed." << std::endl;
    } else {
        std::cerr << "Test failed." << std::endl;
    }
}

int main() {
    run<B2BGemmShape<64 /*M*/, 64 /*N*/, 64 /*K*/, 128 /*P*/>,
        B2BGemmShape<64 /*kTM*/, 64 /*kTN*/, 64 /*kTK*/, 128 /*kTP*/>>();

    run<B2BGemmShape<1024 /*M*/, 128 /*N*/, 128 /*K*/, 128 /*P*/>,
        B2BGemmShape<64 /*kTM*/, 128 /*kTN*/, 128 /*kTK*/, 128 /*kTP*/>>();

    return 0;
}
