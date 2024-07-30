#include "cell/copy/mod.hpp"
#include "cell/sync.hpp"
#include "common/test_utils.hpp"
#include "types/mod.hpp"

#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tiledcuda::testing {
using namespace cell;
using namespace copy;
namespace tl = tile_layout;

#define DEBUG

namespace {
template <typename Element>
__device__ void init_value(Element* data, int numel) {
    for (int i = 0; i < numel; ++i) data[i] = static_cast<Element>(i % 2048);
}

template <typename Element, typename Global, typename Shared1, typename Shared2,
          typename Reg, typename G2S1, typename G2S2, typename S2R>
__global__ void run_test_swizzle(const Element* data, G2S1& copy1, G2S2& copy2,
                                 S2R& copy3) {
    Global g_tile(data);

    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);
    init_value(buf, Shared1::kNumel);
    init_value(buf + Shared1::kNumel, Shared2::kNumel);

    Shared1 s_tile(buf);
    Shared2 s_tile_swizzled(buf + Shared1::kNumel);

    copy1(g_tile, s_tile);
    copy2(g_tile, s_tile_swizzled);
    __copy_async();
    __syncthreads();

    Reg r_tile;
    Reg r_tile_swizzled;

    copy3(s_tile, r_tile);
    copy3(s_tile, r_tile_swizzled);
    __syncthreads();

#if defined(DEBUG)
    if (thread0()) {
        printf("s_tile:\n");
        s_tile.dump_value();

        printf("\ns_tile_swizzled:\n");
        __half* data = reinterpret_cast<__half*>(buf + Shared1::kNumel);
        for (int i = 0; i < Shared2::kNumel; ++i) {
            printf("%.0f, ", __half2float(data[i]));
            if (i && (i + 1) % Shared1::kCols == 0) printf("\n");
        }

        printf("\n\nr_tile:\n");
        r_tile.dump_value();

        printf("\nr_tile_swizzled:\n");
        r_tile_swizzled.dump_value();
    }
#endif

    for (int i = 0; i < Reg::kRows; ++i) {
        for (int j = 0; j < Reg::kCols; ++j) {
            auto data1 = r_tile(i, j).data();
            auto data2 = r_tile_swizzled(i, j).data();

            for (int n = 0; n < Reg::DType::kNumel; ++n) {
                assert(data1[n] == data2[n]);
            }
        }
    }
}

}  // namespace

/*
TEST(TestSwizzledLayout, test1) {
    using Element = cutlass::half_t;

    using WarpLayout = tl::RowMajor<2, 1>;
    const int kThreads = tl::get_numel<WarpLayout> * 32;
    static constexpr int kWarpPerRow = tl::num_rows<WarpLayout>;
    static constexpr int kWarpPerCol = tl::num_cols<WarpLayout>;

    const int kRows = 32;
    const int kCols = 64;

    const int kSc0 = kRows / kWarpPerRow / 16;
    const int kSc1 = kCols / kWarpPerCol / 16;

    // initalize the input matrix
    int numel = kRows * kCols;
    thrust::host_vector<Element> h_A(numel);
    for (int i = 0; i < h_A.size(); ++i) h_A[i] = static_cast<Element>(i);
    thrust::device_vector<Element> d_A = h_A;

    using Global = GlobalTile<Element, tl::RowMajor<kRows, kCols>>;

    using SharedLayout = tl::RowMajor<kRows, kCols>;
    using SwizzledSharedLayout = tl::Swizzled<SharedLayout, 2, 3, 3>;

    using Shared1 = SharedTile<Element, SharedLayout>;
    using Shared2 = SharedTile<Element, SwizzledSharedLayout>;

    using Reg = RegTile<BaseTileRowMajor<Element>, tl::RowMajor<kSc0, kSc1>>;

    using G2S1 = GlobalToSharedLoader<Shared1, WarpLayout>;
    G2S1 copy1;
    using G2S2 = GlobalToSharedLoader<Shared2, WarpLayout>;
    G2S2 copy2;
    using S2R = SharedToRegLoader<Reg, WarpLayout, WarpReuse::kCont>;
    S2R copy3;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);
    int shm_size = (Shared1::kNumel + Shared2::kNumel) * sizeof(Element);

    run_test_swizzle<Element, Global, Shared1, Shared2, Reg, G2S1, G2S2, S2R>
        <<<dim_grid, dim_block, shm_size>>>(
            thrust::raw_pointer_cast(d_A.data()), copy1, copy2, copy3);
    cudaDeviceSynchronize();
}
*/

TEST(TestSwizzledLayout, test2) {
    // unittest for loading gemm's operandA
    using Element = cutlass::half_t;

    using WarpLayout = tl::RowMajor<1, 2>;
    const int kThreads = tl::get_numel<WarpLayout> * 32;
    static constexpr int kWarpPerRow = tl::num_rows<WarpLayout>;
    // static constexpr int kWarpPerCol = tl::num_cols<WarpLayout>;

    const int kRows = 64;
    const int kCols = 64;

    const int kSc0 = kRows / kWarpPerRow / 16;
    const int kSc1 = kCols / 16;

    // initalize the input matrix
    int numel = kRows * kCols;
    thrust::host_vector<Element> h_A(numel);
    for (int i = 0; i < h_A.size(); ++i)
        h_A[i] = static_cast<Element>(i % 2048);
    thrust::device_vector<Element> d_A = h_A;

    using Global = GlobalTile<Element, tl::RowMajor<kRows, kCols>>;

    using SharedLayout = tl::RowMajor<kRows, kCols>;
    using SwizzledSharedLayout = tl::Swizzled<SharedLayout, 2, 3, 3>;

    using Shared1 = SharedTile<Element, SharedLayout>;
    using Shared2 = SharedTile<Element, SwizzledSharedLayout>;

    using Reg = RegTile<BaseTileRowMajor<Element>, tl::RowMajor<kSc0, kSc1>>;

    using G2S1 = GlobalToSharedLoader<Shared1, WarpLayout>;
    G2S1 copy1;
    using G2S2 = GlobalToSharedLoader<Shared2, WarpLayout>;
    G2S2 copy2;
    using S2R = SharedToRegLoader<Reg, WarpLayout, WarpReuse::kRowReuseCont>;
    S2R copy3;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);
    int shm_size = (Shared1::kNumel + Shared2::kNumel) * sizeof(Element);

    run_test_swizzle<Element, Global, Shared1, Shared2, Reg, G2S1, G2S2, S2R>
        <<<dim_grid, dim_block, shm_size>>>(
            thrust::raw_pointer_cast(d_A.data()), copy1, copy2, copy3);
    cudaDeviceSynchronize();
}

}  // namespace tiledcuda::testing
