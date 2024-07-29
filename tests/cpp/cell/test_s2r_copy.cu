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
__device__ bool check_results(const cutlass::half_t* data, int numel) {
    const float epsilon = 1e-3;
    bool pass_test = false;

    for (int i = 0; i < numel; ++i) {
        int v = i % 2048;
        if (abs(data[i] - static_cast<cutlass::half_t>(v)) > epsilon) {
            const __half* ptr = reinterpret_cast<const __half*>(data);
            printf("Error data[%d]; Expected: %d, Got: %.0f\n", i, v,
                   __half2float(ptr[i]));
        }
    }

    pass_test = true;
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
}

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
    memset(buf_, 0, Shared::kNumel * sizeof(DType));  // clean the shared memory

    // the reverse operation, store from register to shared
    storer(r_tile, s_tile);
    __syncthreads();

    if (thread0()) {
        s_tile.dump_value();
        check_results(buf, Shared::kNumel);
    }
}
}  // namespace

/*
TEST(TestShared2Reg, operand_A) {  // load mode for loading operand A in gemm
    using Element = cutlass::half_t;

    using WarpLayout = tl::RowMajor<2, 2>;
    const int kThreads = tl::get_numel<WarpLayout> * 32;

    using Shared = SharedTile<Element, tl::RowMajor<64, 32>>;
    // Each thread accesses 2x4 elements (the shape of `BaseHalfTileRowMajor`)
    // within a 16x16 `BaseTile`. These 2x4 elements are accessed 2x2 times
    // along each dimension, contributing to the final register tile handled by
    // a single thread.
    using Reg = RegTile<BaseTileRowMajor<Element>, tl::RowMajor<2, 2>>;

    // In the `RowReuseCont` mode, warps in the same row repeatedly access the
    // same data.
    using Copy = SharedToRegLoader<Reg, WarpLayout, WarpReuse::kRowReuseCont>;
    Copy copy;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);
    int shm_size = Shared::kNumel * sizeof(Element);

    run_test_load<Element, Shared, Reg, Copy>
        <<<dim_grid, dim_block, shm_size>>>(copy);
    cudaDeviceSynchronize();
}

TEST(TestShared2Reg, operand_B) {  // load mode for loading operand B in gemm
    using Element = cutlass::half_t;

    using WarpLayout = tl::RowMajor<2, 2>;
    const int kThreads = tl::get_numel<WarpLayout> * 32;

    // a 32x64 row-major shared tile is equivalent to a 64x32 col-major tile
    using Shared = SharedTile<Element, tl::RowMajor<32, 64>>;

    // Each thread accesses 4x2 elements (the shape of `BaseHalfTileRowMajor`)
    // within a 16x16 `BaseTile`. These 4x2 elements are accessed 2x2 times
    // along each dimension, contributing to the final register tile handled by
    // a single thread.
    using Reg = RegTile<BaseTileColMajor<Element>, tl::ColMajor<2, 2>>;
    // In the `ColReuseCont` mode, warps in the same column repeatedly access
    // the same data.
    using Copy = SharedToRegLoader<Reg, WarpLayout, WarpReuse::kColReuseCont>;
    Copy copy;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);
    int shm_size = Shared::kNumel * sizeof(Element);

    run_test_load<Element, Shared, Reg, Copy>
        <<<dim_grid, dim_block, shm_size>>>(copy);
    cudaDeviceSynchronize();
}

TEST(TestReg2Shared, operand_C) {
    using Element = cutlass::half_t;

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
*/

TEST(TestShared2Reg, operand_A_swizzle) {
    using Element = __half;

    using WarpLayout = tl::RowMajor<1, 1>;
    const int kThreads = tl::get_numel<WarpLayout> * 32;

    const int kRows = 64;
    const int kCols = 32;

    using SharedLayout = tl::Swizzled<tl::RowMajor<kRows, kCols>, 2, 3, 3>;
    using Shared = SharedTile<Element, SharedLayout>;
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

}  // namespace tiledcuda::testing
