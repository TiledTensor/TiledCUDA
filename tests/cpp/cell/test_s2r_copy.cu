#include "cell/compute/mod.hpp"
#include "cell/copy/mod.hpp"
#include "common/test_utils.hpp"
#include "types/mod.hpp"

#include <glog/logging.h>

namespace tiledcuda::testing {
using namespace cell;
using namespace copy;
namespace tl = tile_layout;

namespace {
template <typename Element>
__device__ void init_value(Element* data, int numel) {
    for (int i = 0; i < numel; ++i) data[i] = static_cast<Element>(i % 2048);
}

template <typename Element>
__device__ bool check_results(const Element* data, int numel);

template <>
__device__ bool check_results(const __half* data, int numel) {
    // const float* data_f = reinterpret_cast<const float*>(data);

    const float epsilon = 1e-4;
    bool pass_test = true;

    float v = 0.;
    float diff = 0.;
    for (int i = 0; i < numel; ++i) {
        v = static_cast<float>(i % 2048);
        diff = abs(__half2float(data[i]) - v);
        if (diff > epsilon) {
            printf("Error data[%d]; Expected: %.0f, Got: %.0f\n", i, v,
                   __half2float(data[i]));
            pass_test = false;
        }
    }

    return pass_test;
}

template <>
__device__ bool check_results(const float* data, int numel) {
    const float epsilon = 1e-4;
    bool pass_test = true;

    for (int i = 0; i < numel; ++i) {
        float v = float(i % 2048);
        if (abs(data[i] - v) > epsilon) {
            printf("Error data[%d]; Expected: %.0f, Got: %.0f\n", i, v,
                   data[i]);
            pass_test = false;
        }
    }

    return pass_test;
}

template <typename Element, typename Shared, typename Reg, typename Copy>
__global__ void run_test_load(Copy& copy) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);
    init_value(buf, Shared::kNumel);

    Shared s_tile(buf);
    Reg r_tile;

    copy(s_tile, r_tile);

#if defined(DEBUG)
    if (thread0()) {
        r_tile.dump_value();
    }
#endif
}

#define DEBUG true

template <typename Shared, typename Reg, typename Loader, typename Storer>
__global__ void run_test_store(Loader& loader, Storer& storer) {
    using DType = typename Shared::DType;

    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<DType*>(buf_);
    init_value(buf, Shared::kNumel);

    Shared s_tile(buf);
    Reg r_tile;

    loader(s_tile, r_tile);  // load from shared to register
    __syncthreads();

#if defined(DEBUG)
    if (thread0()) {
        printf("register tile:\n");
        r_tile.dump_value();
    }
#endif
    memset(buf, 0, Shared::kNumel * sizeof(DType));  // clean the shared memory

    // the reverse operation, store from register to shared
    storer(r_tile, s_tile);
    __syncthreads();

    if (thread0()) {
        s_tile.dump_value();
        assert(check_results(buf, Shared::kNumel));
    }
}

template <typename SharedHalf, typename RegHalf, typename SharedFloat,
          typename RegFloat, typename ConvertHalf, typename Loader,
          typename Storer>
__global__ void run_test_store_float(ConvertHalf& convert, Loader& loader,
                                     Storer& storer) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];

    // load half data to shared memory
    using DType = typename SharedHalf::DType;
    auto* buf = reinterpret_cast<DType*>(buf_);
    init_value(buf, SharedHalf::kNumel);

    // store buffer on shared memory for storing tcu's output register tile
    using AccType = typename SharedFloat::DType;
    auto* store_buf = reinterpret_cast<AccType*>(buf + SharedHalf::kNumel);
    memset(store_buf, 0, SharedFloat::kNumel * sizeof(AccType));

    SharedHalf sh_tile(buf);
    RegHalf rh_tile;
    RegFloat rf_tile;

    loader(sh_tile, rh_tile);  // load from shared to register
    __syncthreads();

    convert(rh_tile, rf_tile);

#if defined(DEBUG)
    if (thread0()) {
        printf("register tile:\n");
        rh_tile.dump_value();

        printf("converted register tile:\n");
        rf_tile.dump_value();
        printf("\n");
    }
#endif

    SharedFloat sf_tile(store_buf);

    // the reverse operation, store from register to shared
    storer(rf_tile, sf_tile);
    __syncthreads();

    if (thread0()) {
        assert(check_results(store_buf, SharedFloat::kNumel));
    }
}
}  // namespace

// TEST(TestShared2Reg, operand_A) {  // load mode for loading operand A in gemm
//     using Element = __half;

//     using WarpLayout = tl::RowMajor<2, 2>;
//     const int kThreads = tl::get_numel<WarpLayout> * 32;

//     using Shared = SharedTile<Element, tl::RowMajor<64, 32>>;
//     // Each thread accesses 2x4 elements (the shape of
//     // `BaseHalfTileRowMajor`)
//     // within a 16x16 `BaseTile`. These 2x4 elements are accessed 2x2 times
//     // along each dimension, contributing to the final register tile handled
//     // by  a single thread.
//     using Reg = RegTile<BaseTileRowMajor<Element>, tl::RowMajor<2, 2>>;

//     // In the `RowReuseCont` mode, warps in the same row repeatedly access
//     //  the same data.
//     using Copy = SharedToRegLoader<Reg, WarpLayout,
//     WarpReuse::kRowReuseCont>; Copy copy;

//     dim3 dim_grid(1, 1, 1);
//     dim3 dim_block(kThreads, 1, 1);
//     int shm_size = Shared::kNumel * sizeof(Element);

//     run_test_load<Element, Shared, Reg, Copy>
//         <<<dim_grid, dim_block, shm_size>>>(copy);
//     cudaDeviceSynchronize();
// }

// TEST(TestShared2Reg, operand_B) {  // load mode for loading operand B in gemm
//     using Element = __half;

//     using WarpLayout = tl::RowMajor<2, 2>;
//     const int kThreads = tl::get_numel<WarpLayout> * 32;

//     // a 32x64 row-major shared tile is equivalent to a 64x32 col-major tile
//     using Shared = SharedTile<Element, tl::RowMajor<32, 64>>;

//     // Each thread accesses 4x2 elements (the shape of
//     // `BaseHalfTileRowMajor`) within a 16x16 `BaseTile`. These 4x2 elements
//     are
//     // accessed 2x2 times along each dimension, contributing to the final
//     // register tile handled by a single thread.
//     using Reg = RegTile<BaseTileColMajor<Element>, tl::ColMajor<2, 2>>;
//     // In the `ColReuseCont` mode, warps in the same column repeatedly access
//     // the same data.
//     using Copy = SharedToRegLoader<Reg, WarpLayout,
//     WarpReuse::kColReuseCont>; Copy copy;

//     dim3 dim_grid(1, 1, 1);
//     dim3 dim_block(kThreads, 1, 1);
//     int shm_size = Shared::kNumel * sizeof(Element);

//     run_test_load<Element, Shared, Reg, Copy>
//         <<<dim_grid, dim_block, shm_size>>>(copy);
//     cudaDeviceSynchronize();
// }

TEST(TestReg2Shared, operand_C) {
    using Element = __half;

    using WarpLayout = tl::RowMajor<1, 1>;
    const int kThreads = tl::get_numel<WarpLayout> * 32;

    using Shared = SharedTile<Element, tl::RowMajor<16, 16>>;
    using Reg = RegTile<BaseTileRowMajor<Element>, tl::RowMajor<1, 1>>;

    using Loader = SharedToRegLoader<Reg, WarpLayout, WarpReuse::kCont>;
    Loader loader;

    using Storer = RegToSharedStorer<Reg, WarpLayout>;
    Storer storer;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);
    int shm_size = Shared::kNumel * sizeof(Element);

    run_test_store<Shared, Reg, Loader, Storer>
        <<<dim_grid, dim_block, shm_size>>>(loader, storer);
    cudaDeviceSynchronize();
}

/*
TEST(TestShared2Reg, operand_A_swizzle) {
    using Element = __half;

    using WarpLayout = tl::RowMajor<1, 1>;
    const int kThreads = tl::get_numel<WarpLayout> * 32;

    const int kRows = 64;
    const int kCols = 32;

    using SharedLayout = tl::RowMajor<kRows, kCols>;
    const bool kUseSwizzledLayout = true;
    using Shared = SharedTile<Element, SharedLayout, kUseSwizzledLayout>;
    using Reg = RegTile<BaseTileRowMajor<Element>, tl::RowMajor<2, 2>>;

    using Copy = SharedToRegLoader<Reg, WarpLayout, WarpReuse::kRowReuseCont>;
    Copy copy;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);
    int shm_size = Shared::kNumel * sizeof(Element);

    run_test_load<Element, Shared, Reg, Copy>
        <<<dim_grid, dim_block, shm_size>>>(copy);
    cudaDeviceSynchronize();
}
*/

/*
TEST(TestReg2Shared, operand_C_float) {
    using Element = __half;
    using AccType = float;

    const int kRowRepeats = 4;
    const int kColRepeats = 8;
    const int kRows = 16 * kRowRepeats;
    const int kCols = 16 * kColRepeats;

    const int kWarpPerRow = 2;
    const int kWarpPerCol = 2;
    using WarpLayout = tl::RowMajor<kWarpPerRow, kWarpPerCol>;
    const int kThreads = tl::get_numel<WarpLayout> * 32;

    using SharedHalf = SharedTile<Element, tl::RowMajor<kRows, kCols>>;
    using RegHalf = RegTile<
        BaseTileRowMajor<Element>,
        tl::RowMajor<kRowRepeats / kWarpPerRow, kColRepeats / kWarpPerCol>>;

    using SharedFloat = SharedTile<AccType, tl::RowMajor<kRows, kCols>>;
    using RegFloat = RegTile<
        BaseTileRowMajor<AccType>,
        tl::RowMajor<kRowRepeats / kWarpPerRow, kColRepeats / kWarpPerCol>>;

    using ConvertHalf = compute::RegTileConvert<RegHalf, RegFloat>;
    ConvertHalf convert;

    using Loader = SharedToRegLoader<RegHalf, WarpLayout, WarpReuse::kCont>;
    Loader loader;

    using Storer = RegToSharedStorer<RegFloat, WarpLayout>;
    Storer storer;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);
    int shm_size = SharedHalf::kNumel * sizeof(Element) +
                   SharedFloat::kNumel * sizeof(AccType);

    run_test_store_float<SharedHalf, RegHalf, SharedFloat, RegFloat,
                         ConvertHalf, Loader, Storer>
        <<<dim_grid, dim_block, shm_size>>>(convert, loader, storer);
    cudaDeviceSynchronize();
}
*/

}  // namespace tiledcuda::testing
