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

template <typename Element, typename WarpLayout, const int kRows,
          const int kCols, const bool kUseSwizzledLayout>
void run_test() {
    static const int kThreads = tl::get_numel<WarpLayout> * 32;

    int numel = kRows * kCols;
    thrust::host_vector<Element> h_A(numel);
    for (int i = 0; i < h_A.size(); ++i)
        h_A[i] = static_cast<Element>(i % 2048);

    thrust::device_vector<Element> d_B(numel);
    thrust::fill(d_B.begin(), d_B.end(), static_cast<Element>(0.));
    thrust::device_vector<Element> d_A = h_A;

    using SrcTile = GlobalTile<Element, tl::RowMajor<kRows, kCols>>;

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

template <typename Element, typename Layout>
__device__ void init_value(Element* buf) {
    Layout layout;
    int count = 0;
    for (int i = 0; i < Layout::kRows; ++i) {
        for (int j = 0; j < Layout::kCols; ++j) {
            buf[layout(i, j)] = static_cast<Element>(++count);
        }
    }
}

template <typename Element, typename SrcTile, typename DstTile, typename Storer>
__global__ void s2g_storer(Element* dst_ptr, Storer& storer) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);

    init_value<Element, DstTile::Layout>(buf);

    SrcTile src(buf);
    DstTile dst(dst_ptr);

    if (thread0()) {
        src.dump_value();
    }
}

template <typename Element, typename WarpLayout, const int kRows,
          const int kCols>
void run_test_storer() {
    static const int kThreads = tl::get_numel<WarpLayout> * 32;

    int numel = kRows * kCols;

    thrust::device_vector<Element> data(numel);
    thrust::fill(data.begin(), data.end(), static_cast<Element>(0.));

    using SrcTile = SharedTile<Element, tl::RowMajor<kRows, kCols>>;
    using DstTile = GlobalTile<Element, tl::RowMajor<kRows, kCols>>;
    using Storer = copy::SharedToGlobalStorer<DstTile, WarpLayout>;
    Storer storer;

    dim3 dim_grid(1, 1);
    dim3 dim_block(kThreads);

    s2g_storer<Element, SrcTile, DstTile, Storer>
        <<<dim_grid, dim_block, numel * sizeof(Element)>>>(
            thrust::raw_pointer_cast(data.data()), storer);
}
}  // namespace

// TEST(GlobalToSharedLoad, test_g2s_loader) {
//     run_test<__half, tl::RowMajor<1, 1>, 16, 16, false>();
//     run_test<__half, tl::RowMajor<1, 1>, 32, 32, false>();
//     run_test<__half, tl::RowMajor<2, 2>, 64, 64, false>();

//     run_test<__half, tl::RowMajor<1, 1>, 16, 16, true>();
//     run_test<__half, tl::RowMajor<1, 1>, 32, 32, true>();
//     run_test<__half, tl::RowMajor<2, 2>, 64, 64, true>();
// }

TEST(SharedToGlobalStore, test_non_swizzled_layout) {
    run_test_storer<float, tl::RowMajor<1, 1>, 16, 16>();
}

}  // namespace tiledcuda::testing
