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

// #define DEBUG

template <typename Reg, typename DType>
DEVICE void check_results(const Reg& r_tile, const Reg& r_tile_swizzled,
                          int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const DType* data1 = r_tile(i, j).data();
            const DType* data2 = r_tile_swizzled(i, j).data();

            for (int n = 0; n < rows * cols; ++n) {
                assert(data1[n] == data2[n]);
            }
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

            if (thread(17)) {
                printf("\niteration [%d, %d]\n", k, i);
                printf("r_tile:\n");
                r_tile.dump_value();

                printf("\nr_tile_swizzled:\n");
                r_tile_swizzled.dump_value();
            }
        }
    }
}

template <typename WarpLayout, const int kRows, const int kCols,
          const int kShmRows, const int kShmCols, const int kChunkShm>
void run_test() {
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
              << "Reg: " << Reg{} << std::endl;
#endif

    using G2S1 = GlobalToSharedLoader<Shared1, WarpLayout>;
    using G2S2 = GlobalToSharedLoader<Shared2, WarpLayout>;
    using S2R = SharedToRegLoader<Reg, WarpLayout, WarpReuse::kRowReuseCont>;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);
    int shm_size = (Shared1::kNumel + Shared2::kNumel) * sizeof(Element);

    // initalize the input matrix
    int numel = kRows * kCols;
    thrust::host_vector<Element> h_A(numel);
    for (int i = 0; i < h_A.size(); ++i)
        h_A[i] = static_cast<Element>(i % 2048);
    thrust::device_vector<Element> d_A = h_A;

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

    test_func<<<dim_grid, dim_block, shm_size>>>(
        thrust::raw_pointer_cast(d_A.data()), g2s, g2s_swizzled, s2r);
    cudaDeviceSynchronize();

    std::ostringstream ss;
    ss << "[" << kRows << ", " << kCols << ", " << kShmRows << ", " << kShmCols
       << ", " << kChunkShm << "]";
    LOG(INFO) << std::endl << ss.str() << " passed!" << std::endl;
}
}  // namespace

TEST(TestSwizzledLayout, test1) {
    run_test<tl::RowMajor<1, 2>, 16, 64, 16, 32, 32>();
    run_test<tl::RowMajor<1, 2>, 16, 128, 16, 64, 32>();
    run_test<tl::RowMajor<1, 2>, 32, 32, 32, 32, 16>();
    run_test<tl::RowMajor<2, 2>, 32, 32, 32, 32, 16>();
    run_test<tl::RowMajor<2, 2>, 32, 32, 32, 32, 32>();
    run_test<tl::RowMajor<2, 2>, 32, 64, 32, 32, 32>();
    run_test<tl::RowMajor<2, 1>, 32, 64, 32, 32, 32>();
    run_test<tl::RowMajor<2, 1>, 32, 128, 32, 64, 32>();
    run_test<tl::RowMajor<2, 1>, 64, 128, 64, 64, 32>();

    // run_test<tl::RowMajor<4, 1>, 64, 64, 64, 64, 64>();

    // run_test<tl::RowMajor<4, 1>, 64, 128, 64, 128, 128>();

    // run_test<tl::RowMajor<4, 1>, 64, 256, 64, 128, 128>();

    // run_test<tl::RowMajor<8, 1>, 128, 512, 128, 256, 128>();
}

}  // namespace tiledcuda::testing
