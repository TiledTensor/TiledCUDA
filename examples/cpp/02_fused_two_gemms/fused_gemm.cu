#include "fused_gemm.hpp"
#include "util.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

template <typename WholeShape, typename CtaTileShape, typename WarpLayout,
          const int kBatch>
void run(std::ofstream& fout, float epsilon = 1e-3, int iters = 20) {
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

    thrust::host_vector<InType> h_a(kM * kK * kBatch);

    for (int i = 0; i < h_a.size(); ++i) {
        h_a[i] = static_cast<InType>(rand_float());
    }

    thrust::host_vector<InType> h_b(kK * kN * kBatch);
    for (int i = 0; i < h_b.size(); ++i) {
        h_b[i] = static_cast<InType>(rand_float());
    }

    thrust::host_vector<InType> h_c(kN * kP * kBatch);
    for (int i = 0; i < h_c.size(); ++i) {
        h_c[i] = static_cast<InType>(rand_float());
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
        FusedGemmTraits<InType, AccType, WholeShape, CtaTileShape, WarpLayout>;

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
    int block_z = kBatch;

    dim3 grid(block_x, block_y, block_z);
    dim3 block(Config::kThreads, 1, 1);

    int shm_input = (kTM * kTK + kTK * kTN + kTN * kTP);
    int shm_output = kTM * kTP;
    int shm_size = shm_input < shm_output ? shm_output * sizeof(InType)
                                          : shm_input * sizeof(InType);

    auto kernel = &KeFusedGemm<InType, AccType,            //
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

    auto err = cudaGetLastError();

    // check err
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                  << " in tiledcuda fused_gemm." << std::endl;
        return;
    }

    h_d = d_d;

    thrust::host_vector<InType> h_acc(kM * kN * kBatch);
    thrust::fill(h_acc.begin(), h_acc.end(), 0.);
    thrust::device_vector<InType> d_acc = h_acc;

    thrust::host_vector<InType> h_d2(kM * kP * kBatch);
    thrust::fill(h_d2.begin(), h_d2.end(), 0.);
    thrust::device_vector<InType> d_d2 = h_d2;

    cublas_two_gemms(kM, kN, kK, kP, kBatch, A, B, C,
                     thrust::raw_pointer_cast(d_d2.data()),
                     thrust::raw_pointer_cast(d_acc.data()), false);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    // check err
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                  << " in cublas two hgemm." << std::endl;
        return;
    }

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

    bool passed = check_results(data, ground_truth, kM * kP * kBatch, epsilon);

    if (passed) {
        std::cout << "[" << kM << ", " << kN << ", " << kK << ", " << kP
                  << "], batch = " << kBatch << ", passed." << std::endl;
    } else {
        std::cout << "[" << kM << ", " << kN << ", " << kK << ", " << kP
                  << "], batch = " << kBatch << ", failed." << std::endl;
    }

    if (passed) {
        int warm_up = 10;
        for (int i = 0; i < warm_up; ++i)
            kernel<<<grid, block, shm_size, 0>>>(A, B, C, D, kM, kN, kK, kP,
                                                 kTM, kTN, kTK, kTP);
        cudaDeviceSynchronize();

        CudaTimer timer;
        timer.start();
        for (int i = 0; i < iters; ++i) {
            kernel<<<grid, block, shm_size, 0>>>(A, B, C, D, kM, kN, kK, kP,
                                                 kTM, kTN, kTK, kTP);
        }
        cudaDeviceSynchronize();
        float time2 = timer.stop() / iters;

        float time1 =
            cublas_two_gemms(kM, kN, kK, kP, kBatch, A, B, C,
                             thrust::raw_pointer_cast(d_d2.data()),
                             thrust::raw_pointer_cast(d_acc.data()), true);

        std::cout << "cuBLAS\tTiledCUDA\tRatio" << std::endl;
        std::cout << std::setprecision(4) << time1 << "\t" << time2 << "\t"
                  << time2 / time1 << std::endl;

        fout << "[" << kM << ", " << kN << ", " << kK << ", " << kP << ", "
             << kBatch << "], " << "\t";
        fout << "[" << kTM << ", " << kTN << ", " << kTK << ", " << kTP << "], "
             << "\t";
        fout << time1 << "\t" << time2 << "\t" << time2 / time1 << std::endl;
    } else {
        std::cerr << "Test failed." << std::endl;
    }
}

int main() {
    std::ofstream fout;
    fout.setf(std::ios::fixed);
    fout.precision(4);

    std::stringstream file_name;
    file_name << "fused_gemm.tsv";
    fout.open(file_name.str(), std::ios::out);

    fout << "[M, N, K, P, kBatch]" << "\t" << "[kTM, kTN, kTK, kTP]" << "\t"
         << "cuBLAS" << "\t" << "TiledCUDA" << "\t" << "Ratio" << std::endl;

    using WarpLayout0 = tl::RowMajor<1, 1>;
    run<FusedGemmShape<16 /*M*/, 16 /*N*/, 16 /*K*/, 16 /*P*/>,
        FusedGemmShape<16 /*kTM*/, 16 /*kTN*/, 16 /*kTK*/, 16 /*kTP*/>,
        WarpLayout0, 1>(fout);

    run<FusedGemmShape<16 /*M*/, 32 /*N*/, 16 /*K*/, 32 /*P*/>,
        FusedGemmShape<16 /*kTM*/, 16 /*kTN*/, 16 /*kTK*/, 32 /*kTP*/>,
        WarpLayout0, 1>(fout);

    using WarpLayout1 = tl::RowMajor<2, 1>;
    run<FusedGemmShape<32 /*M*/, 64 /*N*/, 32 /*K*/, 64 /*P*/>,
        FusedGemmShape<32 /*kTM*/, 32 /*kTN*/, 32 /*kTK*/, 64 /*kTP*/>,
        WarpLayout1, 1>(fout);

    run<FusedGemmShape<64 /*M*/, 64 /*N*/, 32 /*K*/, 64 /*P*/>,
        FusedGemmShape<32 /*kTM*/, 32 /*kTN*/, 32 /*kTK*/, 64 /*kTP*/>,
        WarpLayout1, 1>(fout);

    using WarpLayout2 = tl::RowMajor<4, 1>;

    run<FusedGemmShape<256 /*M*/, 128 /*N*/, 64 /*K*/, 64 /*P*/>,
        FusedGemmShape<64 /*kTM*/, 32 /*kTN*/, 64 /*kTK*/, 64 /*kTP*/>,
        WarpLayout1, 1>(fout, 5e-3);

    run<FusedGemmShape<1024 /*M*/, 1024 /*N*/, 128 /*K*/, 128 /*P*/>,
        FusedGemmShape<64 /*kTM*/, 64 /*kTN*/, 128 /*kTK*/, 128 /*kTP*/>,
        WarpLayout2, 1>(fout, 8e-2 /*epsilon*/);

    // batched
    run<FusedGemmShape<16 /*M*/, 16 /*N*/, 16 /*K*/, 16 /*P*/>,
        FusedGemmShape<16 /*kTM*/, 16 /*kTN*/, 16 /*kTK*/, 16 /*kTP*/>,
        WarpLayout0, 2>(fout);

    run<FusedGemmShape<1024 /*M*/, 1024 /*N*/, 128 /*K*/, 128 /*P*/>,
        FusedGemmShape<64 /*kTM*/, 64 /*kTN*/, 128 /*kTK*/, 128 /*kTP*/>,
        WarpLayout2, 5>(fout, 8e-2 /*epsilon*/);

    run<FusedGemmShape<64 /*M*/, 256 /*N*/, 128 /*K*/, 128 /*P*/>,
        FusedGemmShape<64 /*kTM*/, 64 /*kTN*/, 128 /*kTK*/, 128 /*kTP*/>,
        WarpLayout2, 1>(fout, 8e-2 /*epsilon*/);

    run<FusedGemmShape<2048 /*M*/, 1024 /*N*/, 128 /*K*/, 128 /*P*/>,
        FusedGemmShape<64 /*kTM*/, 128 /*kTN*/, 128 /*kTK*/, 128 /*kTP*/>,
        WarpLayout2, 1>(fout, 8e-2 /*epsilon*/);

    run<FusedGemmShape<1024 /*M*/, 2048 /*N*/, 128 /*K*/, 128 /*P*/>,
        FusedGemmShape<64 /*kTM*/, 128 /*kTN*/, 128 /*kTK*/, 128 /*kTP*/>,
        WarpLayout2, 1>(fout, 8e-2 /*epsilon*/);

    // Test failed.
    // run<FusedGemmShape<2048 /*M*/, 2048 /*N*/, 128 /*K*/, 128 /*P*/>,
    //     FusedGemmShape<64 /*kTM*/, 128 /*kTN*/, 128 /*kTK*/, 128 /*kTP*/>,
    //     WarpLayout2, 1>(fout, 8e-2 /*epsilon*/);

    // CUDA error: misaligned address in tiledcuda fused_gemm.
    // run<FusedGemmShape<2048 /*M*/, 2048 /*N*/, 128 /*K*/, 128 /*P*/>,
    //     FusedGemmShape<128 /*kTM*/, 128 /*kTN*/, 128 /*kTK*/, 128 /*kTP*/>,
    //     WarpLayout2, 1>(fout, 8e-2 /*epsilon*/);

    return 0;
}
