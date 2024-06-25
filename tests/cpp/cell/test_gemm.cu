#include "cell/mod.hpp"
#include "common/test_utils.hpp"
#include "util/debug.hpp"

#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tiledcuda {

using namespace cell;
using namespace cell::copy;
namespace tl = tile_layout;

using namespace cute;

namespace {

template <typename Element, typename ElementAcc, typename LoadSharedA,
          typename LoadSharedB, typename StoreSharedC, typename TileIteratorA,
          typename RegA, typename LoadRegA, typename TileIteratorB,
          typename RegB, typename LoadRegB, typename SharedC, typename RegC>
__global__ void test_wmma1(const Element* ga, const Element* gb, ElementAcc* gc,
                           LoadRegA& load_rA, LoadRegB& load_rB) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* shared_a = reinterpret_cast<Element*>(buf_);
    auto* shared_b = shared_a + TileIteratorA::Tile::kNumel;
    auto* shared_c = reinterpret_cast<ElementAcc*>(buf_);

    // transfer tiles from global to shared memory
    copy_2d_tile_g2s(ga, shared_a, typename LoadSharedA::SrcLayout{},
                     typename LoadSharedA::DstLayout{},
                     typename LoadSharedA::TiledCopy{});
    copy_2d_tile_g2s(gb, shared_b, typename LoadSharedB::SrcLayout{},
                     typename LoadSharedB::DstLayout{},
                     typename LoadSharedB::TiledCopy{});
    __copy_async();
    __syncthreads();

    TileIteratorA sAs(shared_a);
    TileIteratorB sBs(shared_b);

    RegA rA;
    RegB rB;
    RegC acc;

    for (int k = 0; k < TileIteratorA::sc1; ++k) {
        load_rA(sAs(_, k).to_tile(), rA);
        // Since we interpret a column-major B with a shape of [K, N] as a
        // row-major B with a shape of [N, K], here sBs is sliced along the
        // first dimension instead of the second one.
        load_rB(sBs(_, k).to_tile(), rB);

        compute::gemm_(rA, rB, acc);
    }
    __syncthreads();

    SharedC sC(shared_c);
    copy::copy_tile_r2s(acc, sC);
    __syncthreads();

    copy::copy_2d_tile_s2g(shared_c, gc, typename StoreSharedC::SrcLayout{},
                           typename StoreSharedC::DstLayout{},
                           typename StoreSharedC::TiledCopy{});
}

}  // namespace

namespace testing {

TEST(TestWmma, shape1) {
    // M, N, K for shared memory tile
    const int M = 64;
    const int N = 64;
    const int K = 128;

    // unittest for register-level gemm by calling into wmma PTX
    using Element = cutlass::half_t;
    using ElementAcc = float;

    using WarpLayout = tl::RowMajor<2, 2>;
    static const int kThreads = tl::get_numel<WarpLayout> * 32;

    // for global to shared memory copy using CuTe
    using LoadSharedA = traits::G2S2DCopyTraits<Element, M, K, M, K, kThreads,
                                                false /*use swizzle*/>;
    using LoadSharedB = traits::G2S2DCopyTraits<Element, N, K, N, K, kThreads,
                                                false /*use swizzle*/>;

    // transfer operand C from shared memory to global memory
    using StoreSharedC =
        traits::S2G2DCopyTraits<ElementAcc, M, N, M, N, kThreads,
                                false /*use swizzle*/>;

    /*
     *  shared memory tile A [64, 128] is partitioned into a 2D grid of 32x32
     *  sub-tiles:
     *  |--------|----------|----------|----------|----------|
     *  |iterator|    0     |    1     |    2     |    3     |
     *  |--------|----------|----------|----------|----------|
     *  |        |sub-tile 0|sub-tile 1|sub-tile 2|sub-tile 3|
     *  |--------|----------|----------|----------|----------|
     *  |        |sub-tile 4|sub-tile 5|sub-tile 6|sub-tile 7|
     *  |--------|----------|----------|----------|----------|
     */
    using SharedA = SharedTile<Element, tl::RowMajor<M, K>>;
    // [64, 128] is chunked by [32, 32], strip counts = [64/32, 128/32] = [2, 4]
    using TileIteratorA = SharedTileIterator<SharedA, TileShape<32, 32>>;
    using RegA = RegTile<Element, tl::RowMajor<4, 8>>;
    using LoadRegA =
        SharedToRegLoader<RegA, WarpLayout, WarpReuse::RowReuseCont,
                          CopyInst::LoadMat>;
    LoadRegA load_rA;

    LOG(INFO) << "TileIteratorA: [" << TileIteratorA::Tile::kRows << ", "
              << TileIteratorA::Tile::kCols
              << "]; numel = " << TileIteratorA::Tile::kNumel << std::endl
              << "sc0 = " << TileIteratorA::sc0
              << ", sc1 = " << TileIteratorA::sc1 << std::endl;

    /*
     *  shared memory tile B [128, 64] is partitioned into a 2D grid of 32x32
     *  sub-tiles:
     *  |--------|----------|----------|
     *  |iterator|          |          |
     *  |--------|----------|----------|
     *  |   0    |sub-tile 0|sub-tile 1|
     *  |--------|----------|----------|
     *  |   1    |sub-tile 2|sub-tile 3|
     *  |--------|----------|----------|
     *  |   2    |sub-tile 4|sub-tile 5|
     *  |--------|----------|----------|
     *  |   3    |sub-tile 6|sub-tile 7|
     *  |--------|----------|----------|
     */

    // A row-major Tile with a shape of [N, K] is equivalent to a column-major
    // Tile with a shape of [K, N]
    using SharedB = SharedTile<Element, tl::RowMajor<N, K>>;  // 64, 128
    using RegB = RegTile<Element, tl::RowMajor<4, 8>>;
    // [64, 128] is chunked by [32, 32], strip counts = [64/32, 128/32] = [2, 4]
    using TileIteratorB = SharedTileIterator<SharedB, TileShape<32, 32>>;
    using LoadRegB =
        SharedToRegLoader<RegB, WarpLayout, WarpReuse::ColReuseCont,
                          CopyInst::LoadMat>;
    LoadRegB load_rB;

    LOG(INFO) << "TileIteratorB: sc0 = " << TileIteratorB::sc0
              << ", sc1 = " << TileIteratorB::sc1 << std::endl;

    static_assert(TileIteratorA::sc1 == TileIteratorB::sc1,
                  "dimension mismatch!");

    using SharedC = SharedTile<ElementAcc, tl::RowMajor<M, N>>;
    using RegC = RegTile<ElementAcc, tl::RowMajor<2, 8>>;

    int size_ab = (M + N) * K * sizeof(Element);
    int size_c = M * N * sizeof(ElementAcc);
    int shm_size = size_ab > size_c ? size_ab : size_c;

    thrust::host_vector<Element> h_a(M * K);
    thrust::host_vector<Element> h_b(K * N);
    thrust::host_vector<ElementAcc> h_c(M * N);

    for (int i = 0; i < h_a.size(); ++i)
        h_a[i] = static_cast<Element>(i % 2048 / 100.);

    for (int i = 0; i < h_b.size(); ++i)
        h_b[i] = static_cast<Element>(i % 2048 / 100.);

    thrust::fill(h_c.begin(), h_c.end(), 0.);

    thrust::device_vector<Element> d_a = h_a;
    thrust::device_vector<Element> d_b = h_b;
    thrust::device_vector<ElementAcc> d_c = h_c;

    LOG(INFO) << "kThreads: " << kThreads << std::endl;
    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);

    test_wmma1<Element, ElementAcc, LoadSharedA, LoadSharedB, StoreSharedC,
               TileIteratorA, RegA, LoadRegA, TileIteratorB, RegB, LoadRegB,
               SharedC, RegC><<<dim_grid, dim_block, shm_size>>>(
        thrust::raw_pointer_cast(d_a.data()),
        thrust::raw_pointer_cast(d_b.data()),
        thrust::raw_pointer_cast(d_c.data()), load_rA, load_rB);

    cudaDeviceSynchronize();

    thrust::host_vector<float> h_c1;
    h_c1 = d_c;
}

}  // namespace testing
}  // namespace tiledcuda
