#include "cell/copy/mod.hpp"
#include "common/test_utils.hpp"
#include "types/types.hpp"

#include <glog/logging.h>

namespace tiledcuda {

using namespace cell;
using namespace cute;
namespace tl = tile_layout;

namespace testing {

#define EPS 1e-4

namespace {

/// utility function
template <typename Element, const int kNumel>
__device__ void init(Element* data) {
    if (threadIdx.x || blockIdx.x || blockIdx.y) return;

    for (int i = 0; i < kNumel; ++i) {
        int v = i % 2048;
        if (v == 0) {
            // avoid zero. the unittest checks whether the data is
            // copied by checking whether the value loaded into the register
            // file greater than zero.
            data[i] = static_cast<Element>(0.1);
        } else {
            data[i] = static_cast<Element>(v);
        }
    }
}

/// utility function
template <typename Element, const int kRows, const int kCols>
__device__ void print_tile(const Element* data, int delimeter = kCols) {
    if (threadIdx.x || blockIdx.x || blockIdx.y) return;

    for (int i = 0; i < kRows * kCols; ++i) {
        printf("%.0f, ", float(data[i] * 1.0_hf));  // print cutlass::half_t
        if ((i + 1) % delimeter == 0) printf("\n");
    }
    printf("\n");
}

/// @brief 1 warp executes 1 `ldmatrix`
template <typename Element, typename Reg, const int kRows, const int kCols>
__device__ const Element* cute_ldmatrix1(const Element* buf) {
    // this setting increases how many threads are used in a CTA
    using WarpLayout = Layout<Shape<_1, _1, _1>>;
    // this setting increases how many registers are used in a CTA
    using ValueLayout = Tile<_16, _16, _16>;
    // `ldmatrix` loads four 8 x 8 tiles
    using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                              WarpLayout, ValueLayout>;
    using CopyInst = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    __syncthreads();

    TiledMma tiled_mma;
    int tid = threadIdx.x;

    auto row_major = tl::RowMajor<kRows, kCols>{};
    auto tensor = cute::make_tensor(make_smem_ptr(buf), row_major);

    auto tiled_copy = make_tiled_copy_A(CopyInst{}, tiled_mma);
    auto thrd_copy = tiled_copy.get_thread_slice(tid);
    auto src = thrd_copy.partition_S(tensor);  // get shared memory partition

    auto thr_mma = tiled_mma.get_thread_slice(tid);
    // partitioned data tile that is loaded into the current thread's local
    // register file
    auto dst = thr_mma.partition_fragment_A(tensor);
    // make the layout compatible for `tiled_copy`
    auto dst_view = thrd_copy.retile_D(dst);

    for (int i = 0; i < int(size<1>(dst_view)); ++i) {
        for (int j = 0; j < int(size<2>(dst_view)); ++j)
            cute::copy(tiled_copy, src(_, i, j), dst_view(_, i, j));
    }

    int reg_tile_size = int(size(dst_view));
    assert(reg_tile_size == Reg::kNumel);

    return dst.data();
}

/// @brief test 2 warps arranged in 2 x 1 row-major tile executes 4 `ldmatrix`
///        per warp
template <typename Element, typename Reg, const int kRows, const int kCols>
__device__ const Element* cute_ldmatrix2(const Element* buf) {
    // 2 warps along the m dimension.
    // When the stride is not specified for CuTe's layout, the default layout is
    // column-major. For our current macro kernel implementation, we treat warps
    // as being laid out in a row-major fashion. In order to make CuTe's layout
    // for warps also row-major, the stride must be specified.
    using WarpLayout = Layout<Shape<_2, _1, _1>, Stride<_1, _1, _1>>;

    // execute 4 `ldmatrix` along the m and n dimensions
    using ValueLayout = Tile<_32, _16, _32>;
    using CopyInst = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                              WarpLayout, ValueLayout>;  // wmma 16.8.16
    __syncthreads();

    TiledMma tiled_mma;
    int tid = threadIdx.x;

    auto row_major = tl::RowMajor<kRows, kCols>{};
    auto tensor = cute::make_tensor(make_smem_ptr(buf), row_major);

    auto tiled_copy = make_tiled_copy_A(CopyInst{}, tiled_mma);
    auto thrd_copy = tiled_copy.get_thread_slice(tid);
    auto src = thrd_copy.partition_S(tensor);  // get shared memory partition

    auto thr_mma = tiled_mma.get_thread_slice(tid);
    // partitioned data tile loaded into the current thread's local register
    auto dst = thr_mma.partition_fragment_A(tensor);
    // make the layout compatible for `tiled_copy`
    auto dst_view = thrd_copy.retile_D(dst);

    // how many 128-bit registers are used per thread.
    int reg_tile_size = int(size(dst_view));
    assert(reg_tile_size == Reg::kNumel);

    for (int i = 0; i < int(size<1>(dst_view)); ++i) {
        for (int j = 0; j < int(size<2>(dst_view)); ++j)
            cute::copy(tiled_copy, src(_, i, j), dst_view(_, i, j));
    }
    return dst.data();
}

template <typename Element, typename Reg, const int kRows, const int kCols,
          const int kExec>
__device__ const Element* cute_ldmatrix3(const Element* buf) {
    // 2 warps along the m dimension
    // When the stride is not specified for CuTe's layout, the default layout is
    // column-major. For our current macro kernel implementation, we treat warps
    // as being laid out in a row-major fashion. In order to make CuTe's layout
    // for warps also row-major, the stride must be specified.
    using WarpLayout_ = Layout<Shape<_2, _1, _2>, Stride<_2, _1, _1>>;
    // execute 4 `ldmatrix` along the m and n dimensions
    using ValueLayout = Tile<_32, _16, _32>;
    using CopyInst = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                              WarpLayout_, ValueLayout>;  // wmma 16.8.16
    __syncthreads();

    TiledMma tiled_mma;
    int tid = threadIdx.x;

    auto row_major = tl::RowMajor<kRows, kCols>{};
    auto tensor = cute::make_tensor(make_smem_ptr(buf), row_major);

    auto tiled_copy = make_tiled_copy_A(CopyInst{}, tiled_mma);
    auto thrd_copy = tiled_copy.get_thread_slice(tid);
    auto src = thrd_copy.partition_S(tensor);  // get shared memory partition

    auto thr_mma = tiled_mma.get_thread_slice(tid);
    // partitioned data tile loaded into the current thread's local register
    auto dst = thr_mma.partition_fragment_A(tensor);
    // make the layout compatible for `tiled_copy`
    auto dst_view = thrd_copy.retile_D(dst);

    // how many 128-bit registers are used per thread.
    int reg_tile_size = int(size(dst_view));
    assert(reg_tile_size == Reg::kNumel * kExec);

    for (int i = 0; i < int(size<1>(dst_view)); ++i) {
        for (int j = 0; j < int(size<2>(dst_view)); ++j)
            cute::copy(tiled_copy, src(_, i, j), dst_view(_, i, j));
    }
    return dst.data();
}

// unittest for transferring data from shared memory to thread's local register
// file using `ldmatrix`
template <typename Element, typename Shared, typename Reg>
__global__ void test_ldmatrix1() {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);
    init<Element, Shared::kRows * Shared::kCols>(buf);
#ifdef DEBUG
    print_tile<Element, Shared::kRows, Shared::kCols>(buf);
#endif

    auto data = cute_ldmatrix1<Element, Reg, Shared::kRows, Shared::kCols>(buf);

    Shared s_tiles(buf);
    Reg r_tile;
    copy::copy_2d_tile_s2r(s_tiles[make_int2(0, 0)], r_tile);

    const half* data1 = reinterpret_cast<const half*>(data);
    const half* data2 = reinterpret_cast<const half*>(r_tile.data());

    for (int i = 0; i < Reg::kNumel; ++i) {
        float v1 = __half2float(data1[i]);
        float v2 = __half2float(data2[i]);

        assert(v1);
        assert(v2);
        assert(abs(v1 - v2) < EPS);
    }
}

template <typename Element, typename Shared, typename Reg>
__global__ void test_ldmatrix2() {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);
    init<Element, Shared::kRows * Shared::kCols>(buf);

#ifdef DEBUG
    print_tile<Element, Shared::kRows, Shared::kCols>(buf);
#endif

    auto data = cute_ldmatrix2<Element, Reg, Shared::kRows, Shared::kCols>(buf);

    Shared s_tiles(buf);
    Reg r_tile;
    copy::copy_2d_tile_s2r(s_tiles[make_int2(0, 0)], r_tile);
    const half* data2 = reinterpret_cast<const half*>(r_tile.data());

    const half* data1 = reinterpret_cast<const half*>(data);
    for (int i = 0; i < Reg::kNumel; ++i) {
        float v1 = __half2float(data1[i]);
        float v2 = __half2float(data2[i]);

        assert(v1);
        assert(v2);
        assert(abs(v1 - v2) < EPS);
    }
}

template <typename Element, typename Shared, typename Reg>
__global__ void test_ldmatrix3() {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);
    init<Element, Shared::kRows * Shared::kCols>(buf);

#ifdef DEBUG
    print_tile<Element, Shared::kRows, Shared::kCols>(buf);
#endif

    auto data = cute_ldmatrix3<Element, Reg, Shared::kRows, Shared::kCols,
                               Shared::sc0 * Shared::sc1>(buf);
    const half* data1 = reinterpret_cast<const half*>(data);

    Shared s_tiles(buf);
    Reg r_tile;
    const half* data2 = reinterpret_cast<const half*>(r_tile.data());

    for (int i = 0; i < Shared::sc0; ++i) {
        for (int j = 0; j < Shared::sc1; ++j) {
            copy::copy_2d_tile_s2r(s_tiles[make_int2(i, j)], r_tile);

            for (int k = 0; k < Reg::kNumel; ++k) {
                int idx = (i * Shared::sc1 + j) * Reg::kNumel + k;
                float v1 = __half2float(data1[idx]);
                float v2 = __half2float(data2[k]);

                assert(v1);
                assert(v2);
                assert(abs(v1 - v2) < EPS);
            }
        }
    }
}

}  // namespace

TEST(TestShared2Reg, shape1) {
    // the minimum shape that a single warp is used to execute 1 `ldmatrix`.
    using Element = cutlass::half_t;

    // configuration for shared memory tile
    using TemporalExecShared = TileShape<1, 1>;
    using WarpLayout = tl::RowMajor<1, 1>;
    using ThreadLayout = tl::RowMajor<16, 2>;  // fixed when using ldmatrix.
    using ElemDataTileShared = tl::RowMajor<1, 8>;
    // the final copy plan for accessing shared memory
    using Shared = SharedTile<Element, TemporalExecShared, WarpLayout,
                              ThreadLayout, ElemDataTileShared>;

    static_assert(Shared::sc0 == 1 && Shared::sc1 == 1);

    // configuration for register tile
    using TemporalExecReg = TileShape<1, 1>;
    using ElemDataTileReg = tl::RowMajor<1, 8>;  // fixed when using ldmatrix
    // the final copy plan for accessing local register file
    using Reg = RegTile<Element, TemporalExecReg, ElemDataTileReg>;

    const int kThreads = Shared::kThreads;
    LOG(INFO) << std::endl
              << "kThreads: " << kThreads << std::endl
              << "Shared memory tile shape: [" << Shared::kRows << ", "
              << Shared::kCols << "];" << std::endl
              << "Spatial tile shape: [" << Shared::kRowsPerExec << ", "
              << Shared::kColsPerExec << "]; " << std::endl;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);

    int shm_size = Shared::kNumel * sizeof(Element);
    test_ldmatrix1<Element, Shared, Reg><<<dim_grid, dim_block, shm_size>>>();
    cudaDeviceSynchronize();
}

TEST(TestShared2Reg, shape2) {
    // this test case represents using 2 warps arranged as 1 x 2 and each
    // thread executes 4 `ldmatrix` along 2 dimensions.
    using Element = cutlass::half_t;

    // configuration for shared memory tile
    using TemporalExecShared = TileShape<1, 1>;
    using WarpLayout = tl::RowMajor<2, 1>;
    using ThreadLayout = tl::RowMajor<16, 2>;  // fixed when using ldmatrix.
    using ElemDataTileShared = tl::RowMajor<2, 16>;
    // the final copy plan for accessing shared memory
    using Shared = SharedTile<Element, TemporalExecShared, WarpLayout,
                              ThreadLayout, ElemDataTileShared>;

    // configuration for register tile
    using TemporalExecReg = TileShape<2, 2>;
    using ElemDataTileReg = tl::RowMajor<1, 8>;  // fixed when using ldmatrix
    // the final copy plan for accessing local register file
    using Reg = RegTile<Element, TemporalExecReg, ElemDataTileReg>;

    const int kThreads = Shared::kThreads;
    LOG(INFO) << std::endl
              << "kThreads: " << kThreads << std::endl
              << "Shared memory tile shape: [" << Shared::kRows << ", "
              << Shared::kCols << "];" << std::endl
              << "Spatial tile shape: [" << Shared::kRowsPerExec << ", "
              << Shared::kColsPerExec << "]; " << std::endl;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);
    int shm_size = Shared::kNumel * sizeof(Element);
    test_ldmatrix2<Element, Shared, Reg><<<dim_grid, dim_block, shm_size>>>();
    cudaDeviceSynchronize();
}

TEST(TestShared2Reg, shape3) {
    // this test case represents using a single warp is used to execute 4
    // `ldmatrix` along 2 dimensions.
    using Element = cutlass::half_t;

    // configuration for shared memory tile
    using TemporalExecShared = TileShape<2, 1>;
    using WarpLayout = tl::RowMajor<2, 2>;
    using ThreadLayout = tl::RowMajor<16, 2>;  // fixed when using ldmatrix.
    using ElemDataTileShared = tl::RowMajor<2, 16>;
    // the final copy plan for accessing shared memory
    using Shared = SharedTile<Element, TemporalExecShared, WarpLayout,
                              ThreadLayout, ElemDataTileShared>;
    static_assert(Shared::sc0 == 2 && Shared::sc1 == 1);

    // configuration for register tile
    using TemporalExecReg = TileShape<2, 2>;
    using ElemDataTileReg = tl::RowMajor<1, 8>;  // fixed when using ldmatrix
    // the final copy plan for accessing local register file
    using Reg = RegTile<Element, TemporalExecReg, ElemDataTileReg>;

    const int kThreads = Shared::kThreads;
    LOG(INFO) << std::endl
              << "kThreads: " << kThreads << std::endl
              << "Shared memory tile shape: [" << Shared::kRows << ", "
              << Shared::kCols << "];" << std::endl
              << "Spatial tile shape: [" << Shared::kRowsPerExec << ", "
              << Shared::kColsPerExec << "]; " << std::endl;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);
    int shm_size = Shared::kNumel * sizeof(Element);
    test_ldmatrix3<Element, Shared, Reg><<<dim_grid, dim_block, shm_size>>>();
    cudaDeviceSynchronize();
}

}  // namespace testing
}  // namespace tiledcuda
