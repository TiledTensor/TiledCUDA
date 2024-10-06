#include "gemm.hpp"
#include "util.hpp"
#include "util/cuda_timer.hpp"

void run_test() {
    using WholeShape = GemmShape<512, 512, 512>;
    using CtaTileShape = GemmShape<64, 64, 64>;
    using WarpLayout = tl::RowMajor<2, 2>;
    static constexpr int kRK = 32;

    using InType = __half;
    using AccType = float;

    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;

    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;

    thrust::host_vector<InType> h_a(kM * kK);
    for (int i = 0; i < h_a.size(); ++i)
        h_a[i] = static_cast<InType>(rand_float());

    thrust::host_vector<InType> h_b(kK * kN);
    for (int i = 0; i < h_b.size(); ++i)
        h_b[i] = static_cast<InType>(rand_float());

    thrust::host_vector<InType> h_c(kM * kN);
    thrust::fill(h_c.begin(), h_c.end(), 0.);

    thrust::device_vector<InType> d_a = h_a;
    thrust::device_vector<InType> d_b = h_b;
    thrust::device_vector<InType> d_c = h_c;

    const InType* A = thrust::raw_pointer_cast(d_a.data());
    const InType* B = thrust::raw_pointer_cast(d_b.data());
    InType* C = thrust::raw_pointer_cast(d_c.data());

    using Config = KeGemmTraits<InType, AccType, WholeShape, CtaTileShape, kRK,
                                WarpLayout>;
    auto kernel =
        &gemm<InType, AccType, kM, kN, kK, kTM, kTN, kTK,
              typename Config::GIteratorA, typename Config::SIteratorA,
              typename Config::SharedA, typename Config::RegA,
              typename Config::G2SLoaderA, typename Config::S2RLoaderA,
              typename Config::GIteratorB, typename Config::SIteratorB,
              typename Config::SharedB, typename Config::RegB,
              typename Config::G2SLoaderB, typename Config::S2RLoaderB,
              typename Config::GlobalC, typename Config::SharedC,
              typename Config::RegC, typename Config::RegCHalf,
              typename Config::ConvertHalf, typename Config::R2SStorerC,
              typename Config::S2GStorerC>;

    static constexpr int smem_size_inputs = kTK * (kTN + kTM) * sizeof(InType);
    static constexpr int smem_size_accumulators = kTM * kTN * sizeof(AccType);
    static constexpr int smem_size = smem_size_inputs > smem_size_accumulators
                                         ? smem_size_inputs
                                         : smem_size_accumulators;

    const int kMaxSmemPerBlock = 48 * 1024;
    if (smem_size > kMaxSmemPerBlock) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    int block_x = CeilDiv<kM, kTM>;
    int block_y = CeilDiv<kN, kTN>;

    dim3 dim_grid(block_x, block_y, 1);
    dim3 dim_block(Config::kThreads, 1, 1);

    kernel<<<dim_grid, dim_block, smem_size>>>(A, B, C);
    cudaDeviceSynchronize();
    h_c = d_c;

    // check correctness
    thrust::device_vector<InType> d_c2(kM * kN);
    thrust::fill(d_c2.begin(), d_c2.end(), 0.);

    cublas_hgemm(kM, kN, kK, thrust::raw_pointer_cast(d_a.data()),
                 thrust::raw_pointer_cast(d_b.data()),
                 thrust::raw_pointer_cast(d_c2.data()), false /*timeit*/);
    thrust::host_vector<InType> h_c2 = d_c2;

    bool passed = check_results(thrust::raw_pointer_cast(h_c.data()),
                                thrust::raw_pointer_cast(h_c2.data()), kM * kN);

    if (passed) {
        std::cout << "Test passed." << std::endl;

        CudaTimer timer;
        timer.start();
        int iters = 20;
        for (int i = 0; i < iters; ++i) {
            kernel<<<dim_grid, dim_block, smem_size>>>(A, B, C);
        }
        cudaDeviceSynchronize();

        float time1 = cublas_hgemm(
            kM, kN, kK, thrust::raw_pointer_cast(d_a.data()),
            thrust::raw_pointer_cast(d_b.data()),
            thrust::raw_pointer_cast(d_c2.data()), true /*timeit*/);

        float time2 = timer.stop() / iters;
        std::cout << "cuBLAS\tTiledCUDA\tRatio" << std::endl;
        std::cout << std::setprecision(4) << time1 << "\t" << time2 << "\t"
                  << time2 / time1 << std::endl;
    } else {
        std::cerr << "Test failed." << std::endl;
    }
}

int main(int argc, char* argv[]) {
    run_test();
    return 0;
}
