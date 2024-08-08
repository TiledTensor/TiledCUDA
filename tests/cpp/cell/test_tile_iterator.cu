#include "common/test_utils.hpp"
#include "types/mod.hpp"

namespace tiledcuda::testing {

using namespace cell;
namespace tl = tile_layout;

namespace {
template <typename Element, typename Layout>
__device__ void init_value(Element* data, Layout layout) {
    int count = 0;
    for (int i = 0; i < Layout::kRows; ++i) {
        for (int j = 0; j < Layout::kCols; ++j) {
            data[layout(i, j)] = static_cast<Element>(++count);
        }
    }
}

template <typename Iterator>
__global__ void test_tile_iterator() {
    using DType = typename Iterator::Tile::DType;

    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<DType*>(buf_);
    init_value(buf, typename Iterator::Tile::Layout{});

    typename Iterator::Tile s_tile(buf);
    printf("Shared tile:\n");
    s_tile.dump_value();

    Iterator tiles(buf);

    printf("\nIterate over rows.\n");
    for (int i = 0; i < Iterator::sc0; ++i) {
        printf("Iteration-[%d, _]:\n", i);
        tiles(i, _).to_tile().dump_value();
        printf("\n");
    }

    printf("Iterate over columns.\n\n");
    for (int j = 0; j < Iterator::sc1; ++j) {
        printf("Iteration-[_, %d]:\n", j);
        tiles(_, j).to_tile().dump_value();
        printf("\n");
    }

    printf("Iterate over rows and columns.\n\n");
    for (int i = 0; i < Iterator::sc0; ++i) {
        for (int j = 0; j < Iterator::sc1; ++j) {
            printf("Iteration-[%d, %d]:\n", i, j);
            tiles(i, j).dump_value();
            printf("\n");
        }
    }

    printf("Another way to iterate over rows and columns.\n\n");
    for (int i = 0; i < Iterator::sc0; ++i) {
        auto cols = tiles(i, _);

        printf("\n");
        for (int j = 0; j < decltype(cols)::sc1; ++j) {
            printf("Iteration-[%d, %d]:\n", i, j);
            cols(j).dump_value();
        }
    }

    printf("Another way to iterate over rows and columns.\n\n");
    for (int i = 0; i < Iterator::sc1; ++i) {
        auto rows = tiles(_, i);

        for (int j = 0; j < decltype(rows)::sc0; ++j) {
            printf("Iteration-[%d, %d]:\n", i, j);
            rows(j).dump_value();
        }
    }
}
}  // namespace

// FIXME(haruhi): Currently, these unit tests only output the values. Implement
// stricter and more meaningful correctness checks.
TEST(TestTile, test_row_major) {
    using Element = cutlass::half_t;

    const int rows = 4;
    const int cols = 12;

    using Tile = SharedTile<Element, tl::RowMajor<rows, cols>>;
    using Iterator = TileIterator<Tile, TileShape<2, 4>>;

    LOG(INFO) << std::endl << "Test Row-major" << std::endl;

    int shm_size = Tile::kNumel * sizeof(Element);
    // DONOT change this launch config. The unittest is implemented for a
    // single thread.
    test_tile_iterator<Iterator><<<1, 1, shm_size>>>();
    cudaDeviceSynchronize();
}

TEST(TestTile, test_col_major) {
    using Element = cutlass::half_t;

    const int rows = 4;
    const int cols = 12;

    using Tile = SharedTile<Element, tl::ColMajor<rows, cols>>;
    using Iterator = TileIterator<Tile, TileShape<2, 4>>;

    LOG(INFO) << std::endl << "Test Column-major" << std::endl;

    int shm_size = Tile::kNumel * sizeof(Element);
    // DONOT change this launch config. The unittest is implemented for a
    // single thread.
    test_tile_iterator<Iterator><<<1, 1, shm_size>>>();
    cudaDeviceSynchronize();
}

TEST(TestTile, test_swizzled_row_major) {
    using Element = float;

    const int rows = 16;
    const int cols = 16;

    const int chunked_row = 8;
    const int chunked_col = 8;

    using Layout = tl::Swizzled<tl::RowMajor<rows, cols>, 1, 0, 2>;
    using Tile = SharedTile<Element, Layout>;
    using Iterator = TileIterator<Tile, TileShape<chunked_row, chunked_col>>;

    LOG(INFO) << std::endl << "Test Row-major" << std::endl;

    int shm_size = Tile::kNumel * sizeof(Element);
    // DONOT change this launch config. The unittest is implemented for a
    // single thread.
    test_tile_iterator<Iterator><<<1, 1, shm_size>>>();
    cudaDeviceSynchronize();
}

}  // namespace tiledcuda::testing
