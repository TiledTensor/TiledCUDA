#include "cell/copy/global_to_shared_2.hpp"
#include "cell/copy/mod.hpp"
#include "cell/sync.hpp"
#include "common/test_utils.hpp"
#include "types/mod.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tiledcuda::testing {
using namespace cell;

namespace {
template <typename Element>
__device__ bool is_equal(const Element* data1, const Element* data2,
                         int numel) {
    Element epsilon = static_cast<Element>(1e-3);

    for (int i = 0; i < numel; ++i) {
        if (data1[i] - data2[i] > epsilon) {
            printf("%d-th emelemnt is not equal: %.2f, %.2f\n", i,
                   __half2float(data1[i]), __half2float(data2[i]));
            return false;
        }
    }
    return true;
}

template <typename Element, typename GlobalTile, typename SharedTile,
          typename Loader1, typename Loader2>
__global__ void copy_g2s(const Element* src, Loader1& loader1,
                         Loader2& loader2) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf1 = reinterpret_cast<Element*>(buf_);
    auto* buf2 = buf1 + SharedTile::kNumel;

    GlobalTile g_tile(src);
    SharedTile s_tile1(buf1);
    SharedTile s_tile2(buf2);

    loader1(g_tile, s_tile1);
    __copy_async();
    __syncthreads();

    loader2(g_tile, s_tile2);
    __copy_async();
    __syncthreads();

    if (thread(0)) {
        // printf("self-implemented\n");
        // s_tile1.dump_value();

        // printf("cute\n");
        // s_tile2.dump_value();

        assert(is_equal(buf1, buf2, SharedTile::kNumel));
    }
}

template <typename Element, const int kRows, const int kCols,
          typename WarpLayout, bool kSwizzled>
void test_rowmajor() {
    using GlobalTile = GlobalTile<Element, tl::RowMajor<kRows, kCols>>;
    using SharedTile =
        SharedTile<Element, tl::RowMajor<kRows, kCols>, kSwizzled>;

    using Loader1 = copy::GlobalToSharedLoader<SharedTile, WarpLayout>;
    using Loader2 = copy::GlobalToSharedLoader2<SharedTile, WarpLayout>;

    Loader1 loader1;
    Loader2 loader2;

    // threads are arranged as 8 x 4 to perform 2D copy
    static const int kThreads = tl::get_numel<WarpLayout> * 32;

    int numel = kRows * kCols;
    thrust::host_vector<Element> h_A(numel);

    for (int i = 0; i < h_A.size(); ++i) {
        h_A[i] = static_cast<Element>(i);
    }
    thrust::device_vector<Element> d_A = h_A;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads);
    copy_g2s<Element, GlobalTile, SharedTile, Loader1, Loader2>
        <<<dim_grid, dim_block, 2 * kRows * kCols * sizeof(Element)>>>(
            thrust::raw_pointer_cast(d_A.data()), loader1, loader2);
    cudaDeviceSynchronize();
}

template <typename Element, const int kRows, const int kCols,
          typename WarpLayout, bool kSwizzled>
void test_colmajor() {
    using GlobalTile = GlobalTile<Element, tl::ColMajor<kRows, kCols>>;
    using SharedTile =
        SharedTile<Element, tl::ColMajor<kRows, kCols>, kSwizzled>;

    using Loader1 = copy::GlobalToSharedLoader<SharedTile, WarpLayout>;
    using Loader2 = copy::GlobalToSharedLoader2<SharedTile, WarpLayout>;

    Loader1 loader1;
    Loader2 loader2;

    // threads are arranged as 8 x 4 to perform 2D copy
    static const int kThreads = tl::get_numel<WarpLayout> * 32;

    int numel = kRows * kCols;
    thrust::host_vector<Element> h_A(numel);

    for (int i = 0; i < h_A.size(); ++i) {
        h_A[i] = static_cast<Element>(i);
    }
    thrust::device_vector<Element> d_A = h_A;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads);
    copy_g2s<Element, GlobalTile, SharedTile, Loader1, Loader2>
        <<<dim_grid, dim_block, 2 * kRows * kCols * sizeof(Element)>>>(
            thrust::raw_pointer_cast(d_A.data()), loader1, loader2);
    cudaDeviceSynchronize();
}
}  // namespace

TEST(G2S_Copy, load_row_major) {
    test_rowmajor<__half, 128, 64, tl::RowMajor<2, 2>, false>();
    test_rowmajor<__half, 128, 64, tl::RowMajor<2, 2>, true>();
}

TEST(G2S_Copy, load_col_major) {
    test_colmajor<__half, 128, 64, tl::RowMajor<2, 2>, false>();
    test_colmajor<__half, 128, 64, tl::RowMajor<2, 2>, true>();
}

}  // namespace tiledcuda::testing
