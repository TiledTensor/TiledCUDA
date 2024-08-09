#include "cell/copy/mod.hpp"
#include "cell/sync.hpp"
#include "cell/traits/copy.hpp"
#include "common/test_utils.hpp"
#include "types/mod.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tiledcuda::testing {
using namespace cell;

namespace {
template <typename Element, typename SrcTile, typename DstTile, typename Loader,
          typename Storer>
__global__ void copy_g2s(const Element* src_ptr, Element* dst_ptr,
                         Loader& loader, Storer& storer) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);

    SrcTile src(src_ptr);  // global memory tile
    DstTile inter(buf);    // shared memory tile
    SrcTile dst(dst_ptr);  // global memory tile

    loader(src, inter);
    __copy_async();
    __syncthreads();

    storer(inter, dst);
    __copy_async();
    __syncthreads();
}

template <typename WarpLayout, const int kRows, const int kCols>
void run_test() {
    static const int kThreads = tl::get_numel<WarpLayout> * 32;

    using Element = __half;
    int numel = kRows * kCols;
    thrust::host_vector<Element> h_A(numel);
    for (int i = 0; i < h_A.size(); ++i)
        h_A[i] = static_cast<Element>(i % 2048);

    thrust::device_vector<Element> d_B(numel);
    thrust::fill(d_B.begin(), d_B.end(), static_cast<Element>(0.));
    thrust::device_vector<Element> d_A = h_A;

    using SrcTile = GlobalTile<Element, tl::RowMajor<kRows, kCols>>;

    static constexpr bool kUseSwizzledLayout = true;
    using DstTile =
        SharedTile<Element, tl::RowMajor<kRows, kCols>, kUseSwizzledLayout>;

    using Loader = copy::GlobalToSharedLoader<DstTile, WarpLayout>;
    Loader loader;

    using Storer = copy::SharedToGlobalStorer<DstTile, WarpLayout>;
    Storer storer;

    dim3 dim_grid(1, 1);
    dim3 dim_block(kThreads);

    copy_g2s<Element, SrcTile, DstTile, Loader, Storer>
        <<<dim_grid, dim_block, kRows * kCols * sizeof(Element)>>>(
            thrust::raw_pointer_cast(d_A.data()),
            thrust::raw_pointer_cast(d_B.data()), loader, storer);
    cudaDeviceSynchronize();

    // check correctness
    thrust::host_vector<Element> h_B(numel);
    h_B = d_B;

    assert_equal(
        reinterpret_cast<__half*>(thrust::raw_pointer_cast(h_A.data())),
        reinterpret_cast<__half*>(thrust::raw_pointer_cast(h_B.data())), numel);
}
}  // namespace

TEST(GlobalToSharedCopy, test_non_swizzled_layout) {
    run_test<tl::RowMajor<1, 1>, 16, 16>();
    run_test<tl::RowMajor<1, 1>, 32, 32>();
    run_test<tl::RowMajor<2, 2>, 64, 64>();
}

}  // namespace tiledcuda::testing
