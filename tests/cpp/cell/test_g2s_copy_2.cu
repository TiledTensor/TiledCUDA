#include "cell/copy/mod.hpp"
#include "cell/sync.hpp"
#include "cell/traits/copy.hpp"
#include "common/test_utils.hpp"
#include "types/mod.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tiledcuda::testing {
using namespace cell;

template <typename Element, typename SrcTile, typename DstTile, typename Loader,
          typename Storer>
__global__ void copy_g2s(const Element* src_ptr, Element* dst_ptr) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);

    SrcTile src(src_ptr);
    DstTile inter(buf);
    SrcTile dst(dst_ptr);

    Loader loader;
    loader(src, inter);
    __copy_async();
    __syncthreads();

    Storer storer;
    storer(inter, dst);
}

TEST(GlobalToSharedCopy, test1) {
    // The simple test case for 2D copy. Copy a 16x32 matrix from global
    // memory to shared memory using a single warp.
    // NOTE: This unitttest represents the minimum shape and threads in a
    // thread block allowed for a 2D copy operation.
    using Element = __half;

    static constexpr int kRows = 32;
    static constexpr int kCols = 64;

    // initalize the input matrix
    int numel = kRows * kCols;
    thrust::host_vector<Element> h_A(numel);
    for (int i = 0; i < h_A.size(); ++i) h_A[i] = static_cast<Element>(i);

    thrust::device_vector<Element> d_B(numel);
    thrust::fill(d_B.begin(), d_B.end(), static_cast<Element>(0.));
    thrust::device_vector<Element> d_A = h_A;

    using WarpLayout = tl::RowMajor<1, 1>;
    static const int kThreads = tl::get_numel<WarpLayout> * 32;

    // using SharedLayout = tl::RowMajor<kRows, kCols>;
    using SharedLayout =
        tl::Swizzled<tl::RowMajor<kRows, kCols>, 2, 3, 3>::Layout;

    using SrcTile = GlobalTile<Element, tl::RowMajor<kRows, kCols>>;
    using DstTile = SharedTile<Element, SharedLayout>;
    using Loader = copy::GlobalToSharedLoader<DstTile, WarpLayout>;
    using Storer = copy::SharedToGlobalStorer<DstTile, WarpLayout>;

    dim3 dim_grid(1, 1);
    dim3 dim_block(kThreads);

    copy_g2s<Element, SrcTile, DstTile, Loader, Storer>
        <<<dim_grid, dim_block, kRows * kCols * sizeof(Element)>>>(
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
