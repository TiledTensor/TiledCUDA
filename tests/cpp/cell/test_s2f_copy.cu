#include "cell/copy/mod.hpp"
#include "common/test_utils.hpp"
#include "types/types.hpp"

#include <glog/logging.h>

// #define __CUDA_ARCH__ 800

namespace tiledcuda {

using namespace cell;
using namespace cute;
namespace tl = tile_layout;

namespace testing {

namespace {

/// utility function
template <typename Element, const int kNumel>
__device__ void init(Element* data) {
    if (threadIdx.x || blockIdx.x || blockIdx.y) return;

    for (int i = 0; i < kNumel; ++i) data[i] = static_cast<Element>(i % 2048);
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

// unittest for transferring data from shared memory to thread's local register
// file using `ldmatrix`
template <typename Element, typename Shared, typename Reg>
__global__ void test_ldmatrix1() {
    const int kRows = Shared::kRows;
    const int kCols = Shared::kCols;

    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);

    init<Element, kRows * kCols>(buf);

#ifdef DEBUG
    print_tile<Element, kRows, kCols>(buf);
#endif

    Shared s_tiles(buf);
    Reg r_tile;
    copy::copy_2d_tile_s2r(s_tiles[make_int2(0, 0)], r_tile);

    // CuTe's configuration for issue a single ldmatrix
    // this setting increases how many threads are used in a CTA
    using WarpLayout = Layout<Shape<_1, _1, _1>>;
    // this setting increases how many registers are used in a CTA
    using ValueLayout = Tile<_16, _16, _16>;
    // `ldmatrix` loads four 8 x 8 tiles
    using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                              WarpLayout, ValueLayout>;
    using CopyInst = Copy_Atom<SM75_U32x4_LDSM_N, Element>;

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
    auto reg_tile = thr_mma.partition_fragment_A(tensor);
    // make the layout compatible for `tiled_copy`
    auto dst_view = thrd_copy.retile_D(reg_tile);

    for (int i = 0; i < int(size<1>(dst_view)); ++i) {
        for (int j = 0; j < int(size<2>(dst_view)); ++j)
            cute::copy(tiled_copy, src(_, i, j), dst_view(_, i, j));
    }

    int reg_tile_size = int(size(dst_view));
    assert(reg_tile_size == Reg::kNumel);

    const half* data1 = reinterpret_cast<const half*>(reg_tile.data());
    const half* data2 = reinterpret_cast<const half*>(r_tile.data());

    const float epsilon = 1e-3;
    for (int i = 0; i < reg_tile_size; ++i) {
        assert(__half2float(data1[i]) - __half2float(data2[i]) < epsilon);
    }
}

template <typename Element, typename Shared, typename Reg>
__global__ void test_ldmatrix2() {
    const int kRows = Shared::kRows;
    const int kCols = Shared::kCols;

    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);
    init<Element, kRows * kCols>(buf);

#ifdef DEBUG
    print_tile<Element, kRows, kCols>(buf);
#endif

    Shared s_tiles(buf);
    Reg r_tile;
    copy::copy_2d_tile_s2r(s_tiles[make_int2(0, 0)], r_tile);

    // ground-truth, CuTe's ldmatrix
    // this setting increases how many threads are used in a CTA
    using WarpLayout = Layout<Shape<_1, _1, _1>>;
    // this setting increases how many registers are used in a CTA
    using ValueLayout = Tile<_32, _16, _32>;
    // `ldmatrix` loads four 8 x 8 tiles
    using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                              WarpLayout, ValueLayout>;
    using CopyInst = Copy_Atom<SM75_U32x4_LDSM_N, Element>;

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

    // how many 128-bit registers are used per thread.
    int reg_tile_size = int(size(dst_view));
    assert(reg_tile_size == Reg::kNumel);

    const float epsilon = 1e-3;
    for (int i = 0; i < int(size<1>(dst_view)); ++i) {
        for (int j = 0; j < int(size<2>(dst_view)); ++j)
            cute::copy(tiled_copy, src(_, i, j), dst_view(_, i, j));
    }

    const half* data1 = reinterpret_cast<const half*>(dst_view.data());
    const half* data2 = reinterpret_cast<const half*>(r_tile.data());

    for (int i = 0; i < reg_tile_size; ++i) {
        assert(__half2float(data1[i]) - __half2float(data2[i]) < epsilon);
    }
}

#define DEBUG_
template <typename Element, typename Shared, typename Reg>
__global__ void test_ldmatrix3() {
    const int kRows = Shared::kRows;
    const int kCols = Shared::kCols;

    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);
    init<Element, kRows * kCols>(buf);

#ifdef DEBUG
    print_tile<Element, kRows, kCols>(buf);
#endif

    Shared s_tiles(buf);
    Reg r_tile;
    copy::copy_2d_tile_s2r(s_tiles[make_int2(0, 0)], r_tile);

    // ground-truth, CuTe's ldmatrix
    // this setting increases how many threads are used in a CTA
    using WarpLayout = Layout<Shape<_2, _1, _1>>;
    // this setting increases how many registers are used in a CTA
    using ValueLayout = Tile<_32, _16, _32>;
    // `ldmatrix` loads four 8 x 8 tiles
    using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                              WarpLayout, ValueLayout>;

    using CopyInst = Copy_Atom<SM75_U32x4_LDSM_N, Element>;

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

    // how many 128-bit registers are used per thread.
    int reg_tile_size = int(size(dst_view));
    // assert(reg_tile_size == Reg::kNumel);

    for (int i = 0; i < int(size<1>(dst_view)); ++i) {
        for (int j = 0; j < int(size<2>(dst_view)); ++j)
            cute::copy(tiled_copy, src(_, i, j), dst_view(_, i, j));
    }

    const half* data1 = reinterpret_cast<const half*>(dst_view.data());
    const half* data2 = reinterpret_cast<const half*>(r_tile.data());

    if (threadIdx.x == 32) {
        printf("\ntiled mma threads = %d\n", int(size(TiledMma{})));

        printf("reg_tile_size: %d\n", reg_tile_size);
        printf("Reg::kNumel: %d\n", Reg::kNumel);

        printf("\nsrc layout");
        print(layout(src));
        printf("\ndst layout");
        print(layout(dst));
        printf("\ndst_view layout\n");
        print(layout(dst_view));
        printf("\n\n");

        // const float epsilon = 1e-3;
        for (int i = 0; i < reg_tile_size; ++i) {
            // printf("[%d]: %.0f\n", i, __half2float(data1[i]));

            printf("[%d]: %.0f, %0.f\n", i, __half2float(data1[i]),
                   __half2float(data2[i]));

            // assert(__half2float(data1[i]) - __half2float(data2[i]) <
            // epsilon);
        }
        printf("\n");
    }
}

}  // namespace

// TEST(TestShared2Reg, shape1) {
//     // this test case represents the minimum shape that a single warp is used
//     to
//     // execute 1 `ldmatrix`.
//     using Element = cutlass::half_t;

//     // configuration for shared memory tile
//     using TemporalExecShared = TileShape<1, 1>;
//     using WarpLayout = TileShape<1, 1>;
//     using ThreadLayout = TileShape<16, 2>;  // fixed when using ldmatrix.
//     using ElemDataTileShared = TileShape<1, 8>;
//     // the final copy plan for accessing shared memory
//     using Shared = SharedTile<Element, TemporalExecShared, WarpLayout,
//                               ThreadLayout, ElemDataTileShared>;

//     static_assert(Shared::sc0 == 1 && Shared::sc1 == 1);

//     // configuration for register tile
//     using TemporalExecReg = TileShape<1, 1>;
//     using ElemDataTileReg = TileShape<1, 8>;  // fixed when using ldmatrix
//     // the final copy plan for accessing local register file
//     using Reg = RegTile<Element, TemporalExecReg, ElemDataTileReg>;

//     const int kThreads = Shared::kThreads;
//     // LOG(INFO) << "kThreads: " << kThreads << std::endl
//     //           << "Shared memory tile shape: [" << Shared::kRows << ", "
//     //           << Shared::kCols << "];" << std::endl
//     //           << "Spatial tile shape: [" << Shared::kRowsPerExec << ", "
//     //           << Shared::kColsPerExec << "]; " << std::endl;

//     dim3 dim_grid(1, 1, 1);
//     dim3 dim_block(kThreads, 1, 1);

//     int shm_size = Shared::kNumel * sizeof(Element);
//     test_ldmatrix1<Element, Shared, Reg><<<dim_grid, dim_block,
//     shm_size>>>(); cudaDeviceSynchronize();
// }

// TEST(TestShared2Reg, shape2) {
//     // this test case represents using a single warp is used to execute 4
//     // `ldmatrix` along 2 dimensions.
//     using Element = cutlass::half_t;

//     // configuration for shared memory tile
//     using TemporalExecShared = TileShape<1, 1>;
//     using WarpLayout = TileShape<1, 1>;
//     using ThreadLayout = TileShape<16, 2>;  // fixed when using ldmatrix.
//     using ElemDataTileShared = TileShape<2, 16>;
//     // the final copy plan for accessing shared memory
//     using Shared = SharedTile<Element, TemporalExecShared, WarpLayout,
//                               ThreadLayout, ElemDataTileShared>;

//     static_assert(Shared::sc0 == 1 && Shared::sc1 == 1);

//     // configuration for register tile
//     using TemporalExecReg = TileShape<2, 2>;
//     using ElemDataTileReg = TileShape<1, 8>;  // fixed when using ldmatrix
//     // the final copy plan for accessing local register file
//     using Reg = RegTile<Element, TemporalExecReg, ElemDataTileReg>;

//     const int kThreads = Shared::kThreads;
//     dim3 dim_grid(1, 1, 1);
//     dim3 dim_block(kThreads, 1, 1);

//     int shm_size = Shared::kNumel * sizeof(Element);
//     test_ldmatrix2<Element, Shared, Reg><<<dim_grid, dim_block,
//     shm_size>>>(); cudaDeviceSynchronize();
// }

TEST(TestShared2Reg, shape3) {
    // this test case represents using a single warp is used to execute 4
    // `ldmatrix` along 2 dimensions.
    using Element = cutlass::half_t;

    // configuration for shared memory tile
    using TemporalExecShared = TileShape<1, 1>;
    using WarpLayout = TileShape<2, 1>;
    using ThreadLayout = TileShape<16, 2>;  // fixed when using ldmatrix.
    using ElemDataTileShared = TileShape<2, 16>;
    // the final copy plan for accessing shared memory
    using Shared = SharedTile<Element, TemporalExecShared, WarpLayout,
                              ThreadLayout, ElemDataTileShared>;

    static_assert(Shared::sc0 == 1 && Shared::sc1 == 1);

    // configuration for register tile
    using TemporalExecReg = TileShape<2, 2>;
    using ElemDataTileReg = TileShape<1, 8>;  // fixed when using ldmatrix
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
