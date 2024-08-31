#pragma once

#include "types/layout.hpp"
#include "util/debug.hpp"

namespace tiledcuda::cell {
namespace tl = tile_layout;

/// @brief Print a tile of single-precision floating point numbers. NOTE: when
// use print in the device function, do add (if(thread0())) to avoid printing
// multiple times by multiple threads. usage:
// if(thread0()) {
//   print_tile(data, layout);
// }
template <typename Layout>
DEVICE void print_tile(const float* data, const Layout& layout) {
    for (int i = 0; i < tl::num_rows<Layout>; ++i) {
        for (int j = 0; j < tl::num_cols<Layout>; ++j) {
            printf("%.3f, ", data[layout(i, j)]);
        }
        printf("\n");

        if (i && (i + 1) % 16 == 0) printf("\n");
    }
}

/// @brief Print a tile of half-precision floating point numbers.
template <typename Layout>
DEVICE void print_tile(const cutlass::half_t* data, const Layout& layout) {
    const half* data_ = reinterpret_cast<const half*>(data);

    for (int i = 0; i < tl::num_rows<Layout>; ++i) {
        for (int j = 0; j < tl::num_cols<Layout>; ++j) {
            printf("%.3f, ", __half2float(data_[layout(i, j)]));
        }
        printf("\n");

        if (i && (i + 1) % 16 == 0) printf("\n");
    }
}

/// @brief Print a tile of half-precision floating point numbers.
template <typename Layout>
DEVICE void print_tile(const __half* data, const Layout& layout) {
    for (int i = 0; i < tl::num_rows<Layout>; ++i) {
        for (int j = 0; j < tl::num_cols<Layout>; ++j) {
            printf("%.3f, ", __half2float(data[layout(i, j)]));
        }
        printf("\n");

        if (i && (i + 1) % 16 == 0) printf("\n");
    }
}

/// @brief Print a register tile. Since register tile is a nested array-like
///        structure. printing resigter tile hits this function.
template <typename DType, typename Layout>
DEVICE void print_tile(const DType* data, const Layout& layout) {
    for (int i = 0; i < tl::num_rows<Layout>; ++i) {
        for (int j = 0; j < tl::num_cols<Layout>; ++j) {
            auto tile = data[layout(i, j)];
            print_tile(tile.data(), tile.layout());
        }
    }
}

template <typename RegTile>
struct RegVecPrinter {
    static constexpr int kRows = RegTile::kRows;

    DEVICE void operator()(const RegTile& tile, int tid) {
        int lane_id = tid % 32;
        for (int i = 0; i < kRows; ++i) {
            if (lane_id % 4 == 0) {
                printf("%.3f, ", __half2float(tile(i, 0)));
            }
            if (lane_id % 4 == 0) {
                printf("%.3f, ", __half2float(tile(i, 1)));
            }
        }

        if (lane_id == 0) printf("\n");
    }
};

template <typename RegTile, const tl::Layout kLayout>
struct RegTilePrinter {
    constexpr static int kRows = RegTile::kRows;
    constexpr static int kCols = RegTile::kCols;

    void operator()(const RegTile& tile, int tid) {}
};

template <typename RegTile>
struct RegTilePrinter<RegTile, tl::Layout::kRowMajor> {
    constexpr static int kRows = RegTile::kRows;
    constexpr static int kCols = RegTile::kCols;

    using DType = typename RegTile::DType::DType;

    DEVICE void print_tile_col(const RegTile& tile, int lane_id, int row_num,
                               bool is_top) {
        for (int col_num = 0; col_num < kCols; ++col_num) {
            if (is_top) {
                printf("%.3f, %.3f, ",
                       __half2float(tile(row_num, col_num)(0, 0)),
                       __half2float(tile(row_num, col_num)(0, 1)));
                printf("%.3f, %.3f, ",
                       __half2float(tile(row_num, col_num)(1, 0)),
                       __half2float(tile(row_num, col_num)(1, 1)));
            } else {
                printf("%.3f, %.3f, ",
                       __half2float(tile(row_num, col_num)(0, 2)),
                       __half2float(tile(row_num, col_num)(0, 3)));
                printf("%.3f, %.3f, ",
                       __half2float(tile(row_num, col_num)(1, 2)),
                       __half2float(tile(row_num, col_num)(1, 3)));
            }
        }
        if (lane_id % 4 == 0) printf("\n");
    }

    DEVICE void operator()(const RegTile& tile, int tid) {
        // BaseTile base_tile;
        int lane_id = tid % 32;
        for (int i = 0; i < kRows; ++i) {
            // Print top row.
            if (lane_id >= 0 && lane_id <= 3)
                print_tile_col(tile, lane_id, i, true);
            else if (lane_id >= 4 && lane_id <= 7)
                print_tile_col(tile, lane_id, i, true);
            else if (lane_id >= 8 && lane_id <= 11)
                print_tile_col(tile, lane_id, i, true);
            else if (lane_id >= 12 && lane_id <= 15)
                print_tile_col(tile, lane_id, i, true);
            else if (lane_id >= 16 && lane_id <= 19)
                print_tile_col(tile, lane_id, i, true);
            else if (lane_id >= 20 && lane_id <= 23)
                print_tile_col(tile, lane_id, i, true);
            else if (lane_id >= 24 && lane_id <= 27)
                print_tile_col(tile, lane_id, i, true);
            else if (lane_id >= 28 && lane_id <= 31)
                print_tile_col(tile, lane_id, i, true);

            // Print bottom row.
            if (lane_id >= 0 && lane_id <= 3)
                print_tile_col(tile, lane_id, i, false);
            else if (lane_id >= 4 && lane_id <= 7)
                print_tile_col(tile, lane_id, i, false);
            else if (lane_id >= 8 && lane_id <= 11)
                print_tile_col(tile, lane_id, i, false);
            else if (lane_id >= 12 && lane_id <= 15)
                print_tile_col(tile, lane_id, i, false);
            else if (lane_id >= 16 && lane_id <= 19)
                print_tile_col(tile, lane_id, i, false);
            else if (lane_id >= 20 && lane_id <= 23)
                print_tile_col(tile, lane_id, i, false);
            else if (lane_id >= 24 && lane_id <= 27)
                print_tile_col(tile, lane_id, i, false);
            else if (lane_id >= 28 && lane_id <= 31)
                print_tile_col(tile, lane_id, i, false);
        }
        if (lane_id == 0) printf("\n");
    }
};

}  // namespace tiledcuda::cell
