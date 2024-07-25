#include "cell/copy/mod.hpp"
#include "cell/sync.hpp"
#include "cell/traits/copy.hpp"
#include "common/test_utils.hpp"
#include "types/mod.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tiledcuda::testing {

using namespace cell;

template <typename Element, typename GlobalLayout, typename SharedLayout,
          typename WarpLayout>
__global__ void copy_g2s(const Element* src, Element* target) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);

    using Swizzled = tl::Swizzled<SharedLayout, WarpLayout>;

    using SrcTile = GlobalTile<Element, GlobalLayout>;
    using DstTile = SharedTile<Element, SharedLayout>;

    SrcTile src_tile(src);
    DstTile dst_tile(buf);
    SrcTile target_tile(target);

    copy::GlobalToSharedLoader<DstTile, WarpLayout, Swizzled> loader;
    loader(src_tile, dst_tile);

    __copy_async();
    __syncthreads();

    copy::SharedToGlobalStorer<DstTile, WarpLayout, Swizzled> storer;
    storer(dst_tile, target_tile);

    __syncthreads();
}

TEST(TESTG2SharedCopy, copy_2d_tile_g2s) {
    namespace traits = tiledcuda::cell::traits;

    // The simple test case for 2D copy. Copy a 16x32 matrix from global
    // memory to shared memory using a single warp.
    // NOTE: This unitttest represents the minimum shape and threads in a
    // thread block allowed for a 2D copy operation.
    using Element = cutlass::half_t;

    using WarpLayout = tl::RowMajor<1, 1>;

    static constexpr int kRows = 16;
    static constexpr int kCols = 32;

    static constexpr int kShmRows = kRows;
    static constexpr int kShmCols = kCols;

    using GlobalLayout = tl::RowMajor<kRows, kCols>;
    using SharedLayout = tl::RowMajor<kShmRows, kShmCols>;

    // threads are arranged as 8 x 4 to perform 2D copy
    static const int kThreads = tl::get_numel<WarpLayout> * 32;

    int numel = kRows * kCols;
    thrust::host_vector<Element> h_A(numel);

    for (int i = 0; i < h_A.size(); ++i) {
        h_A[i] = __float2half(float(i));
    }

    // copy data from host to device
    thrust::device_vector<Element> d_A = h_A;

    thrust::device_vector<Element> d_B(numel);
    thrust::fill(d_B.begin(), d_B.end(), static_cast<Element>(0.));

    int m = CeilDiv<kRows, kShmRows>;
    int n = CeilDiv<kCols, kShmCols>;
    dim3 dim_grid(m, n);
    dim3 dim_block(kThreads);

    copy_g2s<Element, GlobalLayout, SharedLayout, WarpLayout>
        <<<dim_grid, dim_block, kShmRows * kShmCols * sizeof(Element)>>>(
            thrust::raw_pointer_cast(d_A.data()),
            thrust::raw_pointer_cast(d_B.data()));

    cudaDeviceSynchronize();

    // check correctness
    thrust::host_vector<Element> h_B(numel);
    h_B = d_B;

    assert_equal(
        reinterpret_cast<__half*>(thrust::raw_pointer_cast(h_A.data())),
        reinterpret_cast<__half*>(thrust::raw_pointer_cast(h_B.data())), numel);
}

}  // namespace tiledcuda::testing
