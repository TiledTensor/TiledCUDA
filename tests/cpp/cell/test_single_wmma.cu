#include "cell/mod.hpp"
#include "common/test_utils.hpp"
#include "cuda_utils.hpp"
#include "util/debug.hpp"

#include <glog/logging.h>

namespace tiledcuda {

using namespace cell;
using namespace cell::copy;
namespace tl = tile_layout;

using namespace cute;

namespace {

// In this implementation, A and C are interpreted as being laid out in
// row-major, and B is interpreted as being laid out in column-major.
__device__ void naive_gemm(int kM, int kN, int kK,  //
                           const __half* A, const __half* B, float* C) {
    if (!thread0()) return;

    for (int i = 0; i < kM; ++i) {
        for (int j = 0; j < kN; ++j) {
            float s = 0.;
            for (int k = 0; k < kK; ++k) {
                s += __half2float(A[i * kK + k]) * __half2float(B[k + kK * j]);
            }
            C[i * kN + j] = s;
        }
    }
}

__device__ void check_results(const float* hc1, const float* hc2, int numel) {
    for (int i = 0; i < numel; ++i) assert(abs(hc1[i] - hc2[i]) < 1e-3);

#if defined(DEBUG)
    if (thread0()) {
        printf("\n\nours:\n");
        printf("%d:\t", 0);
        for (int i = 0; i < numel; i++) {
            printf("%.0f, ", hc1[i]);
            if (i & (i + 1) % 32 == 0) printf("\n%d:\t", (i + 1) / 32);
        }
        printf("\nground-truth:\n");
        printf("%d:\t", 0);
        for (int i = 0; i < numel; i++) {
            printf("%.0f, ", hc2[i]);
            if (i & (i + 1) % 32 == 0) printf("\n%d:\t", (i + 1) / 32);
        }
    }
#endif
}

template <typename Element, typename ElementAcc>
__device__ void init_values(Element* a, Element* b, ElementAcc* c, int M, int N,
                            int K) {
    if (!thread0()) return;

    for (int i = 0; i < M * K; ++i) a[i] = static_cast<Element>(i);
    for (int i = 0; i < K * N; ++i) b[i] = static_cast<Element>(i);
    for (int i = 0; i < M * N; ++i) c[i] = 0.;
}

template <typename Element, typename ElementAcc,  //
          const int M, const int N, const int K,  //
          typename TileIteratorA, typename RegA, typename LoadRegA,
          typename TileIteratorB, typename RegB, typename LoadRegB,
          typename SharedC, typename RegC, typename StoreRegC>
__global__ void test_wmma(LoadRegA& load_rA, LoadRegB& load_rB,
                          StoreRegC& store_rC) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* shared_a = reinterpret_cast<Element*>(buf_);
    auto* shared_b = shared_a + TileIteratorA::Tile::kNumel;
    auto* shared_c =
        reinterpret_cast<ElementAcc*>(shared_b + TileIteratorB::Tile::kNumel);

    init_values<Element, ElementAcc>(shared_a, shared_b, shared_c, M, N, K);

    SharedC sC(shared_c);

    TileIteratorA sAs(shared_a);
    TileIteratorB sBs(shared_b);

    RegA rA;
    RegB rB;
    RegC acc;

    for (int k = 0; k < TileIteratorA::sc1; ++k) {
        auto sA = sAs(k);
        auto sB = sBs(k);

        load_rA(sA, rA);
        load_rB(sB, rB);

        compute::gemm_(rA, rB, acc);
    }

    store_rC(acc, sC);
    __syncthreads();

    if (thread0()) {
        __half* dA = reinterpret_cast<__half*>(shared_a);
        __half* dB = reinterpret_cast<__half*>(shared_b);
        float* dC = reinterpret_cast<float*>(shared_c);
        naive_gemm(M, N, K, dA, dB, dC);

        check_results(dC, shared_c, M * N);
    }
}

}  // namespace

namespace testing {

template <typename Element, typename ElementAcc, const int M, const int N,
          const int K>
struct TestTraits {
    using WarpLayout = tl::RowMajor<1, 1>;
    static const int kThreads = tl::get_numel<WarpLayout> * 32;

    // ============= shared to register loader =================
    // how many elements a BaseTile are executed along the m, n, k dimension
    static constexpr int ms = M / 16;
    static constexpr int ns = N / 16;
    static constexpr int ks = K / 16;

    using SharedA = SharedTile<Element, tl::RowMajor<M, K>>;
    using TileIteratorA = SharedTileIterator<SharedA, TileShape<M, K>>;
    using RegA = RegTile<Element, tl::RowMajor<ms * 2, ks * 4>>;
    using LoadRegA =
        SharedToRegLoader<RegA, WarpLayout, WarpReuse::RowReuseCont,
                          CopyInst::LoadMat>;

    using SharedB = SharedTile<Element, tl::ColMajor<K, N>>;
    using RegB = RegTile<Element, tl::RowMajor<ks * 4, ns * 2>>;
    using TileIteratorB = SharedTileIterator<SharedB, TileShape<K, N>>;
    using LoadRegB =
        SharedToRegLoader<RegB, WarpLayout, WarpReuse::ColReuseCont,
                          CopyInst::LoadMat>;

    static_assert(TileIteratorA::sc1 == TileIteratorB::sc0,
                  "dimension mismatch!");

    // ============= register to shared storer =================
    using SharedC = SharedTile<ElementAcc, tl::RowMajor<M, N>>;
    using RegC = RegTile<ElementAcc, tl::RowMajor<ms * 2, ns * 4>>;
    using StoreRegC =
        RegToSharedStorer<RegC, WarpLayout, tl::RegLayout::WMMA_m16n16k16,
                          CopyInst::LoadS32>;
};

template <const int M, const int N, const int K>
void run_test() {
    using Element = cutlass::half_t;
    using ElementAcc = float;

    using config = TestTraits<Element, ElementAcc, M, N, K>;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(config::kThreads, 1, 1);
    int size_ab = (M + N) * K * sizeof(Element);
    int size_c = M * N * sizeof(ElementAcc);
    int shm_size = size_ab + size_c;

    typename config::LoadRegA load_rA;
    typename config::LoadRegB load_rB;
    typename config::StoreRegC store_rC;

    test_wmma<Element, ElementAcc, M, N, K,  //
              typename config::TileIteratorA, typename config::RegA,
              typename config::LoadRegA, typename config::TileIteratorB,
              typename config::RegB, typename config::LoadRegB,
              typename config::SharedC, typename config::RegC,
              typename config::StoreRegC>
        <<<dim_grid, dim_block, shm_size>>>(load_rA, load_rB, store_rC);
    cudaDeviceSynchronize();

    LOG(INFO) << std::endl
              << "[" << M << ", " << N << ", " << K << "]" << std::endl
              << "RegA: [" << config::RegA::kRows << ", " << config::RegA::kCols
              << "]" << std::endl
              << "RegB: [" << config::RegB::kRows << ", " << config::RegB::kCols
              << "]" << std::endl
              << "RegC: [" << config::RegC::kRows << ", " << config::RegC::kCols
              << "]" << std::endl
              << "Test passed!" << std::endl;
}

TEST(TestWmma, test_m16n16k16_f) {
    // this unittest implements using a single warp to execute wmma instructions
    run_test<16, 32, 16>();
    run_test<16, 16, 16>();
    run_test<96, 48, 80>();
    run_test<64, 128, 128>();
}

}  // namespace testing
}  // namespace tiledcuda
