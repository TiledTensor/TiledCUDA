#include "cell/copy/mod.hpp"
#include "common/test_utils.hpp"
#include "types/mod.hpp"

#include <glog/logging.h>

namespace tiledcuda {

using namespace cell;
using namespace copy;

namespace tl = tile_layout;
namespace traits = cell::traits;

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

    if (thread0()) {
        // printf("shared tile:\n");
        // s_tile.dump_value();

        printf("register tile:\n");
        r_tile.dump_value();
    }
}

#define DEBUG
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

namespace testing {

TEST(TestShared2Reg, operand_A) {
    // the load mode for loading operand A in gemm
    using Element = cutlass::half_t;

    using WarpLayout = tl::RowMajor<2, 2>;
    const int kThreads = tl::get_numel<WarpLayout> * 32;

    using Shared = SharedTile<Element, tl::RowMajor<64, 32>>;

    // Execute the 16x16 BaseTile two times along the row dimension and once
    // along the column dimension. The resulting warp tile has a shape of
    // [32, 16].
    // Since there are two warps in a thread block, the shared memory tile shape
    // is [64, 32].
    using Reg = RegTile<BaseHalfTileRowMajor, tl::RowMajor<2, 1>>;

    using Copy = SharedToRegLoader<Reg, WarpLayout, WarpReuse::RowReuseCont>;
    Copy copy;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);
    int shm_size = Shared::kNumel * sizeof(Element);

    run_test_load<Element, Shared, Reg, Copy>
        <<<dim_grid, dim_block, shm_size>>>(copy);
    cudaDeviceSynchronize();
}

TEST(TestShared2Reg, operand_B) {
    // the load mode for loading operand B in gemm
    using Element = cutlass::half_t;

    using WarpLayout = tl::RowMajor<2, 2>;
    const int kThreads = tl::get_numel<WarpLayout> * 32;

    // a 32x64 row-major shared tile is equivalent to a 64x32 col-major tile
    using Shared = SharedTile<Element, tl::RowMajor<32, 64>>;

    // Execute the 16x16 BaseTile two times along the row dimension and once
    // along the column dimension. The resulting warp tile has a shape of
    // [32, 16].
    // Since there are two warps in a thread block, the shared memory tile shape
    // is [64, 32].
    using Reg = RegTile<BaseHalfTileColMajor, tl::ColMajor<2, 1>>;

    using Copy = SharedToRegLoader<Reg, WarpLayout, WarpReuse::ColReuseCont>;
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
    using Reg = RegTile<BaseHalfTileRowMajor, tl::RowMajor<1, 1>>;

    using Loader = SharedToRegLoader<Reg, WarpLayout, WarpReuse::Cont>;
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

}  // namespace testing
}  // namespace tiledcuda
