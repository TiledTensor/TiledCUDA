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

    init_values<Element, ElementAcc>(shared_a, shared_b, shared_c, 16, 16, 16);

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
        naive_gemm(16, 16, 16, dA, dB, dC);

        check_results(dC, shared_c, 16 * 16);
    }
}

}  // namespace

namespace testing {

template <typename Element, typename ElementAcc, typename WarpLayout,
          const int M, const int N, const int K>
struct TestTraits {
    static const int kThreads = tl::get_numel<WarpLayout> * 32;

    // ============= shared to register loader =================
    using SharedA = SharedTile<Element, tl::RowMajor<M, K>>;
    using TileIteratorA = SharedTileIterator<SharedA, TileShape<16, 16>>;
    using RegA = RegTile<Element, tl::RowMajor<2, 4>>;
    using LoadRegA =
        SharedToRegLoader<RegA, WarpLayout, WarpReuse::RowReuseCont,
                          CopyInst::LoadMat>;

    using SharedB = SharedTile<Element, tl::ColMajor<K, N>>;
    using RegB = RegTile<Element, tl::RowMajor<2, 4>>;
    using TileIteratorB = SharedTileIterator<SharedB, TileShape<16, 16>>;
    using LoadRegB =
        SharedToRegLoader<RegB, WarpLayout, WarpReuse::ColReuseCont,
                          CopyInst::LoadMat>;

    static_assert(TileIteratorA::sc1 == TileIteratorB::sc1,
                  "dimension mismatch!");

    // ============= register to shared storer =================
    using SharedC = SharedTile<ElementAcc, tl::RowMajor<M, N>>;
    using RegC = RegTile<ElementAcc, tl::RowMajor<2, 4>>;

    using StoreRegC =
        RegToSharedStorer<RegC, WarpLayout, RegLayout::WMMA_m16n16k16,
                          CopyInst::LoadS32>;
};

void run_test() {
    const int M = 16;
    const int N = 16;
    const int K = 16;

    using WarpLayout = tl::RowMajor<1, 1>;

    using Element = cutlass::half_t;
    using ElementAcc = float;

    using config = TestTraits<Element, ElementAcc, WarpLayout, M, N, K>;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(config::kThreads, 1, 1);
    int size_ab = (M + N) * K * sizeof(Element);
    int size_c = M * N * sizeof(ElementAcc);
    int shm_size = size_ab + size_c;

    typename config::LoadRegA load_rA;
    typename config::LoadRegB load_rB;
    typename config::StoreRegC store_rC;

    test_wmma<Element, ElementAcc,  //
              typename config::TileIteratorA, typename config::RegA,
              typename config::LoadRegA, typename config::TileIteratorB,
              typename config::RegB, typename config::LoadRegB,
              typename config::SharedC, typename config::RegC,
              typename config::StoreRegC>
        <<<dim_grid, dim_block, shm_size>>>(load_rA, load_rB, store_rC);
    cudaDeviceSynchronize();
}

TEST(TestWmma, test_m16n16k16_f) { run_test(); }

}  // namespace testing
}  // namespace tiledcuda
