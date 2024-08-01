#include "cell/compute/softmax.hpp"
#include "cell/copy/constants.hpp"
#include "cell/copy/mod.hpp"
#include "common/test_utils.hpp"
#include "types/mod.hpp"
#include "util/debug.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tiledcuda::testing {
using namespace cell;

template <typename Element>
void cpu_softmax(Element* src, Element* dst, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        Element sum = 0;
        for (int j = 0; j < cols; ++j) {
            sum += exp(src[i * cols + j]);
        }
        for (int j = 0; j < cols; ++j) {
            dst[i * cols + j] = exp(src[i * cols + j]) / sum;
        }
    }
}

template <typename Element, typename RegLayout, typename GlobalLayout,
          typename BaseTile, typename WarpLayout, const tl::Layout kLayout,
          const copy::WarpReuse kMode, const int kHeight, const int kWidth>
__global__ void reg_softmax(Element* src, Element* dst) {
    using SrcLoadTile = GlobalTile<Element, GlobalLayout>;
    using DstLoadTile = RegTile<BaseTile, RegLayout>;
    using SrcReduceTile = DstLoadTile;
    using DstReduceTile = RegTile<Element, tl::RowMajor<kHeight, 2>>;
    using SrcStoreTile = DstLoadTile;
    using DstStoreTile = GlobalTile<Element, GlobalLayout>;

    SrcLoadTile src_load_tile(src);
    DstLoadTile dst_load_tile;
    DstReduceTile dst_reduce_tile;
    DstStoreTile dst_store_tile(dst);

    // Load data from global memory to register file
    copy::GlobalToRegLoader<DstLoadTile, WarpLayout, kMode> loader;
    loader(src_load_tile, dst_load_tile);
    __syncthreads();

    // Execute softmax.
    compute::Softmax<SrcReduceTile, kLayout> row_softmax;
    row_softmax(dst_load_tile, dst_reduce_tile);

    __syncthreads();

    if (thread(0)) {
        printf("Thread 0:\n");
        dst_load_tile.dump_value();
    }

    if (thread(1)) {
        printf("Thread 1:\n");
        dst_load_tile.dump_value();
    }

    if (thread(4)) {
        printf("Thread 4:\n");
        dst_load_tile.dump_value();
    }

    if (thread(8)) {
        printf("Thread 8:\n");
        dst_load_tile.dump_value();
    }

    __syncthreads();
    copy::RegToGlobalStorer<DstStoreTile, SrcStoreTile, WarpLayout> storer;
    storer(dst_load_tile, dst_store_tile);
}

template <typename Element, typename RegLayout, typename GlobalLayout,
          typename BaseTile, typename WarpLayout, const tl::Layout kLayout,
          const copy::WarpReuse kMode, const int kHeight, const int kWidth>
void run_reg_softmax() {
    int kNumel = 16 * 16 * kHeight * kWidth;
    int kWarpSize = tl::get_numel<WarpLayout>;

    srand(42);

    thrust::host_vector<Element> h_src(kNumel);
    for (int i = 0; i < kNumel; ++i) {
        h_src[i] = (Element)(10 * (rand() / float(RAND_MAX)) - 5);
    }

    thrust::device_vector<Element> d_src = h_src;
    thrust::device_vector<Element> d_dst(kNumel);
    thrust::fill(d_dst.begin(), d_dst.end(), static_cast<Element>(0.));

    thrust::host_vector<Element> h_dst_ref(kNumel);
    thrust::fill(h_dst_ref.begin(), h_dst_ref.end(), static_cast<Element>(0.));

    reg_softmax<Element, RegLayout, GlobalLayout, BaseTile, WarpLayout, kLayout,
                kMode, kHeight, kWidth>
        <<<1, 32 * kWarpSize>>>(thrust::raw_pointer_cast(d_src.data()),
                                thrust::raw_pointer_cast(d_dst.data()));

    thrust::host_vector<Element> h_dst = d_dst;

    cpu_softmax(thrust::raw_pointer_cast(h_src.data()),
                thrust::raw_pointer_cast(h_dst_ref.data()), 16 * kHeight,
                16 * kWidth);

    // Check the result
    for (int i = 0; i < kNumel; ++i) {
        EXPECT_NEAR(h_dst[i], h_dst_ref[i], 1e-3);
    }
}

TEST(TestRegSoftmax, row_major_reg_softmax_0) {
    const int kHeight = 1;
    const int kWidth = 1;
    using Element = float;
    using WarpLayout = tl::RowMajor<1, 1>;
    using RegLayout = tl::RowMajor<kHeight, kWidth>;

    const copy::WarpReuse kMode = copy::WarpReuse::kCont;

    using GlobalLayout = tl::RowMajor<16 * kHeight, 16 * kWidth>;

    run_reg_softmax<Element, RegLayout, GlobalLayout, BaseTileRowMajor<Element>,
                    WarpLayout, tl::Layout::kRowMajor, kMode, kHeight,
                    kWidth>();
}

TEST(TestRegSoftmax, row_major_reg_softmax_1) {
    const int kHeight = 2;
    const int kWidth = 2;
    using Element = float;
    using WarpLayout = tl::RowMajor<1, 1>;
    using RegLayout = tl::RowMajor<kHeight, kWidth>;

    const copy::WarpReuse kMode = copy::WarpReuse::kCont;

    using GlobalLayout = tl::RowMajor<16 * kHeight, 16 * kWidth>;

    run_reg_softmax<Element, RegLayout, GlobalLayout, BaseTileRowMajor<Element>,
                    WarpLayout, tl::Layout::kRowMajor, kMode, kHeight,
                    kWidth>();
}

TEST(TestRegSoftmax, row_major_reg_softmax_2) {
    const int kHeight = 4;
    const int kWidth = 1;
    using Element = float;
    using WarpLayout = tl::RowMajor<kHeight, 1>;
    using RegLayout = tl::RowMajor<1, kWidth>;

    const copy::WarpReuse kMode = copy::WarpReuse::kCont;

    using GlobalLayout = tl::RowMajor<16 * kHeight, 16 * kWidth>;

    run_reg_softmax<Element, RegLayout, GlobalLayout, BaseTileRowMajor<Element>,
                    WarpLayout, tl::Layout::kRowMajor, kMode, kHeight,
                    kWidth>();
}

}  // namespace tiledcuda::testing
