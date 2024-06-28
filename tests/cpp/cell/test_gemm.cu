#include "cell/mod.hpp"
#include "common/test_utils.hpp"
#include "cuda_utils.hpp"
#include "util/debug.hpp"

#include <cublas_v2.h>
#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tiledcuda {

using namespace cell;
using namespace cell::copy;
namespace tl = tile_layout;

using namespace cute;

namespace {

// This implementation interprets A and C as being laid out in row-major order,
// while B is laid out in column-major order.
void cublas_hgemm(int m, int n, int k, const __half* A, const __half* B,
                  __half* C, int lda, int ldb, int ldc) {
    __half alf = 1.;
    __half bet = 0.;

    cublasHandle_t handle;
    CublasCheck(cublasCreate(&handle));
    CublasCheck(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alf, A,
                            lda, B, ldb, &bet, C, ldc));
    CublasCheck(cublasDestroy(handle));
}

template <typename Element, typename ElementAcc,                              //
          typename LoadSharedA, typename LoadSharedB, typename StoreSharedC,  //
          typename TileIteratorA, typename RegA, typename LoadRegA,
          typename TileIteratorB, typename RegB, typename LoadRegB,
          typename SharedC, typename RegC, typename StoreRegC>
__global__ void test_wmma1(const Element* ga, const Element* gb, ElementAcc* gc,
                           LoadRegA& load_rA, LoadRegB& load_rB,
                           StoreRegC& store_rC) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* shared_a = reinterpret_cast<Element*>(buf_);
    auto* shared_b = shared_a + TileIteratorA::Tile::kNumel;
    auto* shared_c = reinterpret_cast<ElementAcc*>(buf_);
    SharedC sC(shared_c);

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
        load_rA(sAs(k), rA);
        load_rB(sBs(k), rB);

        compute::gemm_(rA, rB, acc);
    }
    __syncthreads();

    store_rC(acc, sC);
    __syncthreads();

    copy_2d_tile_s2g(shared_c, gc, typename StoreSharedC::SrcLayout{},
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
     *  shared memory tile A [64, 128] is partitioned into a 2D grid of 64 x 32
     *  sub-tiles:
     *  |--------|----------|----------|----------|----------|
     *  |iterator|    0     |    1     |    2     |    3     |
     *  |--------|----------|----------|----------|----------|
     *  |        |sub-tile 0|sub-tile 1|sub-tile 2|sub-tile 3|
     *  |--------|----------|----------|----------|----------|
     */
    using SharedA = SharedTile<Element, tl::RowMajor<M, K>>;
    // [64, 128] is chunked by [32, 32], strip counts = [64/64, 128/32] = [1, 4]
    using TileIteratorA = SharedTileIterator<SharedA, TileShape<64, 32>>;
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
     *  shared memory tile B [128, 64] is partitioned into a 2D grid of 32 x 64
     *  sub-tiles:
     *  |--------|----------|
     *  |iterator|          |
     *  |--------|----------|
     *  |   0    |sub-tile 0|
     *  |--------|----------|
     *  |   1    |sub-tile 2|
     *  |--------|----------|
     *  |   2    |sub-tile 4|
     *  |--------|----------|
     *  |   3    |sub-tile 6|
     *  |--------|----------|
     */

    // A row-major Tile with a shape of [N, K] is equivalent to a column-major
    // Tile with a shape of [K, N]
    using SharedB = SharedTile<Element, tl::RowMajor<N, K>>;  // 64, 128
    using RegB = RegTile<Element, tl::RowMajor<4, 8>>;
    // [64, 128] is chunked by [32, 32], strip counts = [64/64, 128/32] = [1, 4]
    using TileIteratorB = SharedTileIterator<SharedB, TileShape<64, 32>>;
    using LoadRegB =
        SharedToRegLoader<RegB, WarpLayout, WarpReuse::ColReuseCont,
                          CopyInst::LoadMat>;
    LoadRegB load_rB;

    LOG(INFO) << "TileIteratorB: sc0 = " << TileIteratorB::sc0
              << ", sc1 = " << TileIteratorB::sc1 << std::endl;

    static_assert(TileIteratorA::sc1 == TileIteratorB::sc1,
                  "dimension mismatch!");

    using SharedC = SharedTile<ElementAcc, tl::RowMajor<M, N>>;  // 64, 64
    using RegC = RegTile<ElementAcc, tl::RowMajor<4, 8>>;

    using StoreRegC =
        RegToSharedStorer<RegC, WarpLayout, RegLayout::WMMA_m16n16k16,
                          CopyInst::LoadS32>;
    StoreRegC store_rC;

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
               SharedC, RegC, StoreRegC><<<dim_grid, dim_block, shm_size>>>(
        thrust::raw_pointer_cast(d_a.data()),
        thrust::raw_pointer_cast(d_b.data()),
        thrust::raw_pointer_cast(d_c.data()), load_rA, load_rB, store_rC);
    cudaDeviceSynchronize();

    {
        // unittest for correctness
        // cublas as the ground-truth
        thrust::device_vector<__half> d_cublas(M * N);
        thrust::fill(d_cublas.begin(), d_cublas.end(), 0.);

        // Matrix A has a row-major layout with dimensions [M, K],
        // Matrix B has a column-major layout with dimensions [K, N],
        // and Matrix C has a row-major layout with dimensions [M, N].
        //
        // This is equivalent to the following:
        // Matrix A has a column-major layout with dimensions [K, M],
        // Matrix B has a column-major layout with dimensions [K, N],
        // and Matrix C has a column-major layout with dimensions [N, M].
        // cuBlas is a Fortran-style(column-major) BLAS library,
        // then we compute: C = B^T @ A
        //             [N, M] = [N, K] @ [K, M]
        cublas_hgemm(N, M, K,
                     reinterpret_cast<const __half*>(
                         thrust::raw_pointer_cast(d_b.data())),
                     reinterpret_cast<const __half*>(
                         thrust::raw_pointer_cast(d_a.data())),
                     reinterpret_cast<__half*>(
                         thrust::raw_pointer_cast(d_cublas.data())),
                     K /*lda*/, K /*ldb*/, N /*ldc*/);

        h_c = d_c;
        thrust::host_vector<__half> h_groundtruth = d_cublas;
    }
}

}  // namespace testing
}  // namespace tiledcuda
