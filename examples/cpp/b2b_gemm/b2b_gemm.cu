#include "util.hpp"

// #define DEBUG

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
    const InType* A = dA + blockIdx.z * (kM * kK) + blockIdx.x * (kTM * kK);
    const InType* B = dB + blockIdx.z * (kK * kN);
    const InType* gC_ptr =
        dC + blockIdx.z * (kN * kP) + blockIdx.y * (kTP * kN);

    AccType* gD_ptr = dD + blockIdx.z * (kM * kP) + blockIdx.x * (kTM * kP) +
                      (blockIdx.y * kTP);

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

    // for (int n = 0; n < GIteratorC::sc0; ++n) {
    for (int n = 0; n < 1; ++n) {
        load_sc(gCs(n), sC);

        if (tiledcuda::thread0()) {
            printf("\niter %d\ngC:\n", n);

            gCs(n).dump_value();

            printf("sC:\n");
            sC.dump_value();
        }

        // for (int k = 0; k < GIteratorA::sc1; ++k) {
        //     load_sa(gAs(k), sA);
        //     load_sb(gBs(k, n), sB);
        //     __copy_async();
        //     __syncthreads();

        //     load_ra(sA, rA);
        //     load_rb(sB, rB);
        //     __syncthreads();

        //     compute::gemm_(rA, rB, acc);
        // }
        // load_rc(sC, rC);
        // __syncthreads();

        // ConvertAcc cast_acc;  // Convert acc to half precision
        // cast_acc(acc, acc_half);

        // compute::gemm_(acc_half, rC, rD);
        // acc.clear();
    }
    __syncthreads();

    GlobalD gD(gD_ptr);
    DStorer storer_d;  // Store D tile from register to global.
    storer_d(rD, gD);
}

template <typename WholeShape, typename CtaTileShape, typename WarpLayout,
          const int kBatch>
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

    static_assert(kK == kTK, "The current implementation requires kTK == K.");
    static_assert(kP == kTP, "The current implementation requires kTP == P.");

    // initialize data
    thrust::host_vector<InType> h_a(kM * kK * kBatch);

    for (int i = 0; i < h_a.size(); ++i) {
        // h_a[i] = static_cast<InType>(rand_float());
        h_a[i] = static_cast<InType>(i);
    }

    thrust::host_vector<InType> h_b(kK * kN * kBatch);
    for (int i = 0; i < h_b.size(); ++i) {
        // h_b[i] = static_cast<InType>(rand_float());
        h_b[i] = static_cast<InType>(i);
    }

    thrust::host_vector<InType> h_c(kN * kP * kBatch);
    for (int i = 0; i < h_c.size(); ++i) {
        // h_c[i] = static_cast<InType>(rand_float());
        h_c[i] = static_cast<InType>(i);
    }

    thrust::host_vector<AccType> h_d(kM * kP * kBatch);
    thrust::fill(h_d.begin(), h_d.end(), 0.);

    thrust::device_vector<InType> d_a = h_a;
    thrust::device_vector<InType> d_b = h_b;
    thrust::device_vector<InType> d_c = h_c;
    thrust::device_vector<AccType> d_d = h_d;

    const InType* A = thrust::raw_pointer_cast(d_a.data());
    const InType* B = thrust::raw_pointer_cast(d_b.data());
    const InType* C = thrust::raw_pointer_cast(d_c.data());
    AccType* D = thrust::raw_pointer_cast(d_d.data());

    using Config =
        B2BGemmTraits<InType, AccType, WholeShape, CtaTileShape, WarpLayout>;

    using RegA = typename Config::RegA;
    using RegB = typename Config::RegB;
    using RegC = typename Config::RegC;
    using RegD = typename Config::RegD;
    using RegAcc = typename Config::RegAcc;
    using RegAccCast = typename Config::RegAccCast;

    std::cout << "RegA: " << RegA{} << std::endl
              << "RegB: " << RegB{} << std::endl
              << "RegC: " << RegC{} << std::endl
              << "RegD: " << RegD{} << std::endl
              << "RegAcc: " << RegAcc{} << std::endl
              << "RegAccCast: " << RegAccCast{} << std::endl;

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
    int block_z = kBatch;

    dim3 grid(block_x, block_y, block_z);
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
    cudaDeviceSynchronize();

    h_d = d_d;

    thrust::host_vector<InType> h_acc(kM * kN * kBatch);
    thrust::fill(h_acc.begin(), h_acc.end(), 0.);
    thrust::device_vector<InType> d_acc = h_acc;

    thrust::host_vector<InType> h_d2(kM * kP * kBatch);
    thrust::fill(h_d2.begin(), h_d2.end(), 0.);
    thrust::device_vector<InType> d_d2 = h_d2;

    cublas_two_gemms(kM, kN, kK, kP, kBatch, A, B, C,
                     thrust::raw_pointer_cast(d_d2.data()),
                     thrust::raw_pointer_cast(d_acc.data()));
    cudaDeviceSynchronize();
    h_acc = d_acc;
    h_d2 = d_d2;

    float* data = thrust::raw_pointer_cast(h_d.data());
    __half* ground_truth = thrust::raw_pointer_cast(h_d2.data());

#ifdef DEBUG
    printf("ours:\n");
    for (int i = 0; i < h_d.size(); ++i) {
        printf("%.3f, ", data[i]);
        if (i && (i + 1) % 16 == 0) printf("\n");
    }
    printf("\nground_truth:\n");
    for (int i = 0; i < h_d.size(); ++i) {
        printf("%.3f, ", __half2float(ground_truth[i]));
        if (i && (i + 1) % 16 == 0) printf("\n");
    }
#endif

    if (check_results(data, ground_truth, kM * kP)) {
        std::cout << "Test passed." << std::endl;
    } else {
        std::cerr << "Test failed." << std::endl;
    }
}

int main() {
    using WarpLayout0 = tl::RowMajor<1, 1>;
    using WarpLayout1 = tl::RowMajor<2, 1>;
    using WarpLayout2 = tl::RowMajor<4, 1>;

    // run<B2BGemmShape<32 /*M*/, 64 /*N*/, 32 /*K*/, 16 /*P*/>,
    //     B2BGemmShape<32 /*kTM*/, 32 /*kTN*/, 32 /*kTK*/, 16 /*kTP*/>,
    //     WarpLayout1, 1>();  // ok

    // run<B2BGemmShape<64 /*M*/, 64 /*N*/, 32 /*K*/, 16 /*P*/>,
    //     B2BGemmShape<32 /*kTM*/, 32 /*kTN*/, 32 /*kTK*/, 16 /*kTP*/>,
    //     WarpLayout1, 1>();  // failed

    // run<B2BGemmShape<16 /*M*/, 16 /*N*/, 16 /*K*/, 16 /*P*/>,
    //     B2BGemmShape<16 /*kTM*/, 16 /*kTN*/, 16 /*kTK*/, 16 /*kTP*/>,
    //     WarpLayout0, 1>();  // ok

    // run<B2BGemmShape<16 /*M*/, 32 /*N*/, 16 /*K*/, 16 /*P*/>,
    //     B2BGemmShape<16 /*kTM*/, 16 /*kTN*/, 16 /*kTK*/, 16 /*kTP*/>,
    //     WarpLayout0, 1>();  // ok

    // run<B2BGemmShape<16 /*M*/, 16 /*N*/, 16 /*K*/, 32 /*P*/>,
    //     B2BGemmShape<16 /*kTM*/, 16 /*kTN*/, 16 /*kTK*/, 32 /*kTP*/>,
    //     WarpLayout0, 1>();  // ok

    run<B2BGemmShape<16 /*M*/, 32 /*N*/, 16 /*K*/, 32 /*P*/>,
        B2BGemmShape<16 /*kTM*/, 16 /*kTN*/, 16 /*kTK*/, 32 /*kTP*/>,
        WarpLayout0, 1>();  // failed

    // run<B2BGemmShape<32 /*M*/, 64 /*N*/, 32 /*K*/, 16 /*P*/>,
    //     B2BGemmShape<32 /*kTM*/, 32 /*kTN*/, 32 /*kTK*/, 16 /*kTP*/>,
    //     WarpLayout1, 1>();

    // run<B2BGemmShape<64 /*M*/, 64 /*N*/, 32 /*K*/, 16 /*P*/>,
    //     B2BGemmShape<64 /*kTM*/, 32 /*kTN*/, 32 /*kTK*/, 16 /*kTP*/>,
    //     WarpLayout1, 1>();

    // run<B2BGemmShape<64 /*M*/, 64 /*N*/, 64 /*K*/, 16 /*P*/>,
    //     B2BGemmShape<64 /*kTM*/, 32 /*kTN*/, 64 /*kTK*/, 16 /*kTP*/>,
    //     WarpLayout1, 1>();

    // run<B2BGemmShape<64 /*M*/, 64 /*N*/, 64 /*K*/, 64 /*P*/>,
    //     B2BGemmShape<64 /*kTM*/, 32 /*kTN*/, 64 /*kTK*/, 64 /*kTP*/>,
    //     WarpLayout1, 1>();

    // run<B2BGemmShape<64 /*M*/, 64 /*N*/, 64 /*K*/, 64 /*P*/>,
    //     B2BGemmShape<64 /*kTM*/, 32 /*kTN*/, 64 /*kTK*/, 64 /*kTP*/>,
    //     WarpLayout1, 1>();  // failed

    // run<B2BGemmShape<1024 /*M*/, 128 /*N*/, 128 /*K*/, 128 /*P*/>,
    //     B2BGemmShape<64 /*kTM*/, 128 /*kTN*/, 128 /*kTK*/, 128
    //     /*kTP*/>, 2>();

    return 0;
}
