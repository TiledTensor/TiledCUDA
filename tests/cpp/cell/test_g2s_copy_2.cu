#include "cell/copy/global_to_shared_2.hpp"
#include "cell/copy/mod.hpp"
#include "cell/sync.hpp"
#include "common/test_utils.hpp"
#include "types/mod.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tiledcuda::testing {
using namespace cell;
using namespace copy;

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
#if defined(DEBUG)
        printf("self-implemented\n");
        s_tile1.dump_value();

        printf("cute\n");
        s_tile2.dump_value();
#endif

        assert(is_equal(buf1, buf2, SharedTile::kNumel));
    }
}

template <typename Element, typename Global, typename Reg, typename Shared,
          typename Loader, typename StorerR2S, typename StorerS2G>
__global__ void store_s2g(const Element* src, Element* dst) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);

    Loader loader;
    StorerR2S storer1;
    StorerS2G storer2;

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

#if defined(DEBUG)
    if (thread0()) {
        printf("\nglobal tile source:\n");
        g_src_tile.dump_value();

        printf("\nshared tile:\n");
        s_tile.dump_value();

        printf("\nglobal tile target:\n");
        g_dst_tile.dump_value();
    }
#endif
}

template <typename Element, const int kRows, const int kCols,
          typename WarpLayout, bool kSwizzled>
void test_row_major_load() {
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
void test_col_major_load() {
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

template <typename Element, const int kRows, const int kCols,
          typename WarpLayout, const bool kSwizzled>
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
    using StorerS2G = SharedToGlobalStorer2<Shared, WarpLayout>;

    int numel = kRows * kCols;
    thrust::host_vector<Element> h_src(numel);
    for (int i = 0; i < h_src.size(); ++i) h_src[i] = static_cast<Element>(i);

    thrust::device_vector<Element> d_src = h_src;

    thrust::device_vector<Element> d_dst(numel);
    thrust::fill(d_dst.begin(), d_dst.end(), static_cast<Element>(0.));

    auto test_func =
        &store_s2g<Element, Global, Reg, Shared, Loader, StorerR2S, StorerS2G>;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);
    int shm_size = Shared::kNumel * sizeof(Element);

    test_func<<<dim_grid, dim_block, shm_size>>>(
        thrust::raw_pointer_cast(d_src.data()),
        thrust::raw_pointer_cast(d_dst.data()));
    cudaDeviceSynchronize();

    thrust::host_vector<Element> h_dst = d_dst;

    assert_equal(thrust::raw_pointer_cast(h_src.data()),
                 thrust::raw_pointer_cast(h_dst.data()), numel, 1e-4);

    LOG(INFO) << "[" << kRows << ", " << kCols << "] test passed!" << std::endl;
};
}  // namespace

TEST(G2S_Copy, load_row_major) {
    test_row_major_load<__half, 128, 64, tl::RowMajor<2, 2>, false>();
    test_row_major_load<__half, 128, 64, tl::RowMajor<2, 2>, true>();
}

TEST(G2S_Copy, load_col_major) {
    test_col_major_load<__half, 128, 64, tl::RowMajor<2, 2>, false>();
    test_col_major_load<__half, 128, 64, tl::RowMajor<2, 2>, true>();
}

TEST(S2G_Copy, store_row_major) {
    test_row_major_store<__half, 128, 64, tl::RowMajor<2, 2>, false>();
    test_row_major_store<__half, 128, 64, tl::RowMajor<2, 2>, true>();

    test_row_major_store<cutlass::half_t, 128, 64, tl::RowMajor<2, 2>, false>();
    test_row_major_store<cutlass::half_t, 128, 64, tl::RowMajor<2, 2>, true>();
}
}  // namespace tiledcuda::testing
