#include "cell/copy/mod.hpp"
#include "cell/sync.hpp"
#include "common/test_utils.hpp"
#include "types/mod.hpp"

#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <sstream>

namespace tiledcuda::testing {
using namespace cell;
using namespace copy;
namespace tl = tile_layout;

namespace {
template <typename Element>
__device__ void init_value(Element* data, int numel) {
    for (int i = 0; i < numel; ++i) {
        data[i] = static_cast<Element>(0.);
    }
}

__global__ void init_halfs(__half* data, int64_t numel) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numel) {
        data[tid] = __float2half(tid % 2048);
    }
}

template <typename Reg, typename DType>
DEVICE void check_results(const Reg& r_tile, const Reg& r_tile_swizzled,
                          int rows, int cols) {
    const int numel = BaseTileRowMajor<DType>::kNumel;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const DType* data1 = r_tile(i, j).data();
            const DType* data2 = r_tile_swizzled(i, j).data();

            for (int n = 0; n < numel; ++n) assert(data1[n] == data2[n]);
        }
    }
}

template <typename Element, typename Global, typename GIterator,
          typename Shared1, typename SIterator1, typename Shared2,
          typename SIterator2, typename Reg, typename G2S1, typename G2S2,
          typename S2R>
__global__ void swizzled_copy(const Element* data, G2S1& g2s,
                              G2S2& g2s_swizzled, S2R& s2r) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);
    init_value(buf, Shared1::kNumel + Shared2::kNumel);

    GIterator g_tiles(data);

    Shared1 s_tile(buf);
    Shared2 s_swizzled_tile(buf + Shared1::kNumel);

    Reg r_tile;
    Reg r_tile_swizzled;

    SIterator1 s_tiles(buf);
    SIterator2 s_swizzled_tiles(buf + Shared1::kNumel);

    for (int k = 0; k < GIterator::sc1; ++k) {
        g2s(g_tiles(k), s_tile);
        g2s_swizzled(g_tiles(k), s_swizzled_tile);
        __copy_async();
        __syncthreads();

        for (int i = 0; i < SIterator1::sc1; ++i) {
            s2r(s_tiles(i), r_tile);
            s2r(s_swizzled_tiles(i), r_tile_swizzled);
            __syncthreads();

            check_results<Reg, Element>(r_tile, r_tile_swizzled, Reg::kRows,
                                        Reg::kCols);
#ifdef DEBUG
            if (thread(0)) {
                printf("\niteration [%d, %d]\n", k, i);
                printf("r_tile:\n");
                r_tile.dump_value();

                printf("\nr_tile_swizzled:\n");
                r_tile_swizzled.dump_value();
            }
#endif
        }
    }
}

/// @brief This unit test verifies the correctness of the swizzled row-major
///        format for loading operand A in GEMM.
template <typename WarpLayout, const int kRows, const int kCols,
          const int kShmRows, const int kShmCols, const int kChunkShm>
void run_test_rowmajor() {
    static_assert(kShmRows == kRows, "kShmRows must be equal to kRows");

    using Element = __half;
    const int kThreads = tl::get_numel<WarpLayout> * 32;
    static constexpr int kWarpPerRow = tl::num_rows<WarpLayout>;

    using Global = GlobalTile<Element, tl::RowMajor<kRows, kCols>>;
    using GIterator = TileIterator<Global, TileShape<kRows, kShmCols>>;

    // for non-swizzled layout
    using Shared1 = SharedTile<Element, tl::RowMajor<kShmRows, kShmCols>,
                               false /*enable swizzled layout on shared*/>;
    using SIterator1 = TileIterator<Shared1, TileShape<kShmRows, kChunkShm>>;

    // for swizzled layout
    using Shared2 = SharedTile<Element, tl::RowMajor<kShmRows, kShmCols>,
                               true /*enable swizzled layout on shared*/>;
    using SIterator2 = TileIterator<Shared2, TileShape<kShmRows, kChunkShm>>;

    using BaseShape = traits::BaseTileShape<Element>;

    const int kSc0 = kShmRows / kWarpPerRow / BaseShape::kRows;
    const int kSc1 = kChunkShm / BaseShape::kCols;

    using Reg = RegTile<BaseTileRowMajor<Element>, tl::RowMajor<kSc0, kSc1>>;

#ifdef DEBUG
    LOG(INFO) << "GIterator: " << GIterator{} << std::endl
              << "SIterator1: " << SIterator1{} << std::endl
              << "SIterator2: " << SIterator2{} << std::endl
              << "GlobalTile Shape: [" << kRows << ", " << kCols << "]"
              << std::endl
              << "SharedTile Shape: [" << kShmRows << ", " << kShmCols << "]"
              << std::endl
              << "sc0: " << kSc0 << ", sc1: " << kSc1 << std::endl
              << "RegTile Shape: " << Reg{} << std::endl;
#endif

    using G2S1 = GlobalToSharedLoader<Shared1, WarpLayout>;
    using G2S2 = GlobalToSharedLoader<Shared2, WarpLayout>;
    using S2R = SharedToRegLoader<Reg, WarpLayout, WarpReuse::kRowReuseCont>;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);
    int shm_size = (Shared1::kNumel + Shared2::kNumel) * sizeof(Element);

    const int numel = kRows * kCols;
    __half* dA;
    CudaCheck(cudaMalloc(&dA, numel * sizeof(__half)));
    const int threads = 128;
    const int blocks = CeilDiv<numel, threads>;
    init_halfs<<<blocks, threads>>>(dA, numel);

    G2S1 g2s;
    G2S2 g2s_swizzled;
    S2R s2r;

    auto test_func =
        &swizzled_copy<Element, Global, GIterator, Shared1, SIterator1, Shared2,
                       SIterator2, Reg, G2S1, G2S2, S2R>;

    // maximal statically allocated smem per block
    const int kMaxSmemPerBlock = 48 * 1024;
    if (shm_size > kMaxSmemPerBlock) {
        cudaFuncSetAttribute(
            test_func, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    }

    test_func<<<dim_grid, dim_block, shm_size>>>(dA, g2s, g2s_swizzled, s2r);
    cudaDeviceSynchronize();

    CudaCheck(cudaFree(dA));

    std::ostringstream ss;
    ss << "[" << kRows << ", " << kCols << ", " << kShmRows << ", " << kShmCols
       << ", " << kChunkShm << "]";
    LOG(INFO) << std::endl << ss.str() << " passed!" << std::endl;
}

/// @brief This unit test verifies the correctness of the swizzled column-major
///        format for loading operand B in GEMM.
template <typename WarpLayout, const int kRows /*K*/, const int kCols /*N*/,
          const int kShmRows, const int kShmCols, const int kChunkShm>
void run_test_colmajor() {
    using Element = __half;
    const int kThreads = tl::get_numel<WarpLayout> * 32;
    static constexpr int kWarpPerCol = tl::num_cols<WarpLayout>;

    static_assert(kShmCols == kCols, "kShmCols must be equal to kCols.");

    using Global = GlobalTile<Element, tl::ColMajor<kRows, kCols>>;
    using GIterator = TileIterator<Global, TileShape<kShmRows, kShmCols>>;

    // for non-swizzled layout
    using Shared1 = SharedTile<Element, tl::ColMajor<kShmRows, kShmCols>,
                               false /*enable swizzled layout on shared*/>;
    using SIterator1 = TileIterator<Shared1, TileShape<kChunkShm, kShmCols>>;

    // for swizzled layout
    using Shared2 = SharedTile<Element, tl::ColMajor<kShmRows, kShmCols>,
                               true /*enable swizzled layout on shared*/>;
    using SIterator2 = TileIterator<Shared2, TileShape<kChunkShm, kShmCols>>;

    using BaseShape = traits::BaseTileShape<Element>;

    const int kSc0 = kChunkShm / BaseShape::kRows;
    const int kSc1 = kShmCols / BaseShape::kCols / kWarpPerCol;

    using Reg = RegTile<BaseTileColMajor<Element>, tl::ColMajor<kSc0, kSc1>>;

#ifdef DEBUG
    LOG(INFO) << std::endl
              << "GIterator: " << GIterator{} << std::endl
              << "SIterator1: " << SIterator1{} << std::endl
              << "SIterator2: " << SIterator2{} << std::endl
              << "GlobalTile Shape: [" << kRows << ", " << kCols << "]"
              << std::endl
              << "SharedTile Shape: [" << kShmRows << ", " << kShmCols << "]"
              << std::endl
              << "sc0: " << kSc0 << ", sc1: " << kSc1 << std::endl
              << "RegTile Shape: " << Reg{} << std::endl;
#endif

    using G2S1 = GlobalToSharedLoader<Shared1, WarpLayout>;
    using G2S2 = GlobalToSharedLoader<Shared2, WarpLayout>;
    using S2R = SharedToRegLoader<Reg, WarpLayout, WarpReuse::kColReuseCont>;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);
    int shm_size = (Shared1::kNumel + Shared2::kNumel) * sizeof(Element);

    const int numel = kRows * kCols;
    __half* dA;
    CudaCheck(cudaMalloc(&dA, numel * sizeof(__half)));
    const int threads = 128;
    const int blocks = CeilDiv<numel, threads>;
    init_halfs<<<blocks, threads>>>(dA, numel);

    G2S1 g2s;
    G2S2 g2s_swizzled;
    S2R s2r;

    auto test_func =
        &swizzled_copy<Element, Global, GIterator, Shared1, SIterator1, Shared2,
                       SIterator2, Reg, G2S1, G2S2, S2R>;

    // maximal statically allocated smem per block
    const int kMaxSmemPerBlock = 48 * 1024;
    if (shm_size > kMaxSmemPerBlock) {
        cudaFuncSetAttribute(
            test_func, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    }

    test_func<<<dim_grid, dim_block, shm_size>>>(dA, g2s, g2s_swizzled, s2r);
    cudaDeviceSynchronize();

    CudaCheck(cudaFree(dA));

    std::ostringstream ss;
    ss << "[" << kRows << ", " << kCols << ", " << kShmRows << ", " << kShmCols
       << ", " << kChunkShm << "]";
    LOG(INFO) << std::endl << ss.str() << " passed!" << std::endl;
}

#define DEBUG

template <typename Element, typename Global, typename Reg, typename Shared,
          typename Loader, typename StorerR2S, typename StorerS2G>
__global__ void swizzled_store(const Element* src, Element* dst, Loader loader,
                               StorerR2S storer1, StorerS2G storer2) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);

    Global g_src_tile(src);
    Reg r_tile;

    Shared s_tile(buf);
    Global g_dst_tile(dst);

    loader(g_src_tile, r_tile);
    __syncthreads();

    storer1(r_tile, s_tile);
    __syncthreads();

    storer2(s_tile, g_dst_tile);
    __syncthreads();

#ifdef DEBUG
    if (thread0()) {
        printf("\nreg tile:\n");
        r_tile.dump_value();

        printf("\nshared tile:\n");
        s_tile.dump_value();

        // printf("\nglobal tile:\n");
        // g_dst_tile.dump_value();
    }
#endif
}

template <typename Element, typename WarpLayout, const int kRows,
          const int kCols, const bool kSwizzled>
void test_row_major_store() {
    using BaseShape = traits::BaseTileShape<Element>;

    const int kThreads = tl::get_numel<WarpLayout> * 32;

    // define tiles
    using Global = GlobalTile<Element, tl::RowMajor<kRows, kCols>>;
    static constexpr int kRowRepeats =
        kRows / tl::num_rows<WarpLayout> / BaseShape::kTileSize;
    static constexpr int kColRepeats =
        kCols / tl::num_cols<WarpLayout> / BaseShape::kTileSize;

    using Reg = RegTile<BaseTileRowMajor<Element>,
                        tl::RowMajor<kRowRepeats, kColRepeats>>;
    using Shared = SharedTile<Element, tl::RowMajor<kRows, kCols>, kSwizzled>;

    // define loader and storer
    using Loader = GlobalToRegLoader<Reg, WarpLayout, copy::WarpReuse::kCont>;
    using StorerR2S = RegToSharedStorer<Reg, WarpLayout>;
    using StorerS2G = SharedToGlobalStorer<Shared, WarpLayout>;

    Loader loader;
    StorerR2S storer1;
    StorerS2G storer2;

    int numel = kRows * kCols;
    thrust::host_vector<Element> h_src(numel);
    for (int i = 0; i < h_src.size(); ++i) {
        h_src[i] = static_cast<Element>(i);
    }
    thrust::device_vector<Element> d_src = h_src;

    thrust::device_vector<Element> d_dst(numel);
    thrust::fill(d_dst.begin(), d_dst.end(), static_cast<Element>(0.));

    auto test_func = &swizzled_store<Element, Global, Reg, Shared, Loader,
                                     StorerR2S, StorerS2G>;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);
    int shm_size = Shared::kNumel * sizeof(Element);

    test_func<<<dim_grid, dim_block, shm_size>>>(
        thrust::raw_pointer_cast(d_src.data()),
        thrust::raw_pointer_cast(d_dst.data()), loader, storer1, storer2);
    cudaDeviceSynchronize();

    thrust::host_vector<Element> h_dst = d_dst;

    assert_equal(thrust::raw_pointer_cast(h_src.data()),
                 thrust::raw_pointer_cast(h_dst.data()), numel, 1e-4);
};

template <typename Element, typename WarpLayout, const int kRows,
          const int kCols, const bool kSwizzled>
void test_col_major_store() {
    using BaseShape = traits::BaseTileShape<Element>;
    const int kThreads = tl::get_numel<WarpLayout> * 32;

    // define tiles
    using Global = GlobalTile<Element, tl::ColMajor<kRows, kCols>>;
    static constexpr int kRowRepeats =
        kRows / tl::num_rows<WarpLayout> / BaseShape::kTileSize;
    static constexpr int kColRepeats =
        kCols / tl::num_cols<WarpLayout> / BaseShape::kTileSize;
    using Reg = RegTile<BaseTileColMajor<Element>,
                        tl::ColMajor<kRowRepeats, kColRepeats>>;
    using Shared = SharedTile<Element, tl::ColMajor<kRows, kCols>, kSwizzled>;

    // define loader and storer
    using Loader = GlobalToRegLoader<Reg, WarpLayout, copy::WarpReuse::kCont>;
    using StorerR2S = RegToSharedStorer<Reg, WarpLayout>;
    using StorerS2G = SharedToGlobalStorer<Shared, WarpLayout>;

    Loader loader;
    StorerR2S storer1;
    StorerS2G storer2;

    int numel = kRows * kCols;
    thrust::host_vector<Element> h_src(numel);
    for (int i = 0; i < h_src.size(); ++i) {
        h_src[i] = static_cast<Element>(i);
    }
    thrust::device_vector<Element> d_src = h_src;

    thrust::device_vector<Element> d_dst(numel);
    thrust::fill(d_dst.begin(), d_dst.end(), static_cast<Element>(0.));

    auto test_func = &swizzled_store<Element, Global, Reg, Shared, Loader,
                                     StorerR2S, StorerS2G>;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);
    int shm_size = Shared::kNumel * sizeof(Element);

    test_func<<<dim_grid, dim_block, shm_size>>>(
        thrust::raw_pointer_cast(d_src.data()),
        thrust::raw_pointer_cast(d_dst.data()), loader, storer1, storer2);
    cudaDeviceSynchronize();

    thrust::host_vector<Element> h_dst = d_dst;

    assert_equal(thrust::raw_pointer_cast(h_src.data()),
                 thrust::raw_pointer_cast(h_dst.data()), numel, 1e-4);
};
}  // namespace

// TEST(TestSwizzledLayout, test_load_row_major) {
//     run_test_rowmajor<tl::RowMajor<1, 2>, 16, 64, 16, 32, 32>();
//     run_test_rowmajor<tl::RowMajor<1, 2>, 16, 128, 16, 64, 32>();
//     run_test_rowmajor<tl::RowMajor<1, 2>, 32, 32, 32, 32, 16>();

//     run_test_rowmajor<tl::RowMajor<2, 2>, 32, 32, 32, 32, 16>();
//     run_test_rowmajor<tl::RowMajor<2, 2>, 32, 32, 32, 32, 32>();
//     run_test_rowmajor<tl::RowMajor<2, 2>, 128, 256, 128, 128, 64>();

//     run_test_rowmajor<tl::RowMajor<2, 1>, 32, 64, 32, 32, 32>();
//     run_test_rowmajor<tl::RowMajor<2, 1>, 32, 128, 32, 64, 32>();
//     run_test_rowmajor<tl::RowMajor<2, 1>, 64, 256, 64, 128, 64>();
//     run_test_rowmajor<tl::RowMajor<4, 1>, 64, 64, 64, 64, 32>();
//     run_test_rowmajor<tl::RowMajor<4, 1>, 64, 128, 64, 64, 64>();
//     run_test_rowmajor<tl::RowMajor<4, 1>, 128, 64, 128, 64, 64>();
//     run_test_rowmajor<tl::RowMajor<4, 1>, 64, 64, 64, 64, 64>();
//     run_test_rowmajor<tl::RowMajor<4, 1>, 64, 128, 64, 128, 128>();
//     run_test_rowmajor<tl::RowMajor<4, 1>, 64, 256, 64, 128, 128>();
//     run_test_rowmajor<tl::RowMajor<8, 1>, 128, 512, 128, 256, 128>();
// }

// TEST(TestSwizzledLayout, test_load_col_major) {
//     run_test_colmajor<tl::RowMajor<1, 1>, 16 /*K*/, 16 /*N*/, 16, 16, 16>();
//     run_test_colmajor<tl::RowMajor<1, 1>, 64 /*K*/, 64 /*N*/, 32, 64, 16>();
//     run_test_colmajor<tl::RowMajor<1, 2>, 128 /*K*/, 32 /*N*/, 64, 32, 32>();
//     run_test_colmajor<tl::RowMajor<2, 1>, 256 /*K*/, 64 /*N*/, 128, 64,
//     32>(); run_test_colmajor<tl::RowMajor<2, 2>, 256 /*K*/, 128 /*N*/, 64,
//     128, 32>(); run_test_colmajor<tl::RowMajor<4, 1>, 128 /*K*/, 64 /*N*/,
//     64, 64, 64>();
// }

// TEST(TestNonSwizzledStore, test_row_major) {
//     static constexpr int kSwizzled = false;
//     test_row_major_store<float, tl::RowMajor<1, 1>, 16, 16, kSwizzled>();
//     test_row_major_store<float, tl::RowMajor<2, 1>, 64, 32, kSwizzled>();
//     test_row_major_store<float, tl::RowMajor<1, 2>, 128, 64, kSwizzled>();
//     test_row_major_store<float, tl::RowMajor<2, 2>, 64, 64, kSwizzled>();

//     test_row_major_store<__half, tl::RowMajor<1, 1>, 16, 16, kSwizzled>();
//     test_row_major_store<__half, tl::RowMajor<2, 1>, 64, 32, kSwizzled>();
//     test_row_major_store<__half, tl::RowMajor<1, 2>, 128, 64, kSwizzled>();
//     test_row_major_store<__half, tl::RowMajor<2, 2>, 64, 64, kSwizzled>();
// }

TEST(TestSwizzledStored, test_row_major) {
    static constexpr int kSwizzled = true;
    /*
    // bank conflict free
    // test_row_major_store<float, tl::RowMajor<1, 1>, 16, 16, kSwizzled>();
    // bank conflict free
    test_row_major_store<float, tl::RowMajor<1, 1>, 16, 48, kSwizzled>();
    // bank conflict free
    test_row_major_store<float, tl::RowMajor<2, 1>, 32, 48, kSwizzled>();

    // FIXME(haruhi): below test cases have bank conflicts. In the current
    // implementation, a single `BaseTile` store/load shared memory will cause 8
    // bank conflicts.

    // This test case has 32 bank conflicts in total
    test_row_major_store<float, tl::RowMajor<1, 1>, 16, 32, kSwizzled>();
    // This test case has 128 bank conflicts in total
    test_row_major_store<float, tl::RowMajor<2, 1>, 64, 32, kSwizzled>();
    // This test case has 512 bank conflicts in total
    test_row_major_store<float, tl::RowMajor<1, 2>, 128, 64, kSwizzled>();
    // This test case has 256 bank conflicts in total
    test_row_major_store<float, tl::RowMajor<2, 2>, 64, 64, kSwizzled>();
    */

    test_row_major_store<__half, tl::RowMajor<1, 1>, 16, 32, kSwizzled>();
    // test_row_major_store<__half, tl::RowMajor<2, 2>, 64, 64, kSwizzled>();
}

// TEST(TestNonSwizzledStored, test_col_major) {
//     static constexpr int kSwizzled = false;
//     test_col_major_store<__half, tl::RowMajor<1, 1>, 16, 16, kSwizzled>();
//     test_row_major_store<__half, tl::RowMajor<2, 1>, 64, 32, kSwizzled>();
//     test_row_major_store<__half, tl::RowMajor<1, 2>, 128, 64, kSwizzled>();
//     test_row_major_store<__half, tl::RowMajor<2, 2>, 64, 64, kSwizzled>();

//     test_col_major_store<float, tl::RowMajor<1, 1>, 16, 16, kSwizzled>();
//     test_col_major_store<float, tl::RowMajor<2, 1>, 64, 32, kSwizzled>();
//     test_col_major_store<float, tl::RowMajor<1, 2>, 128, 64, kSwizzled>();
//     test_col_major_store<float, tl::RowMajor<2, 2>, 64, 64, kSwizzled>();
// }

// TEST(TestSwizzledStored, test_col_major) {
//     static constexpr int kSwizzled = true;
//     test_col_major_store<float, tl::RowMajor<1, 1>, 16, 16, kSwizzled>();
//     test_col_major_store<float, tl::RowMajor<2, 1>, 64, 32, kSwizzled>();
//     test_col_major_store<float, tl::RowMajor<1, 2>, 128, 64, kSwizzled>();
//     test_col_major_store<float, tl::RowMajor<2, 2>, 64, 64, kSwizzled>();
// }

}  // namespace tiledcuda::testing
