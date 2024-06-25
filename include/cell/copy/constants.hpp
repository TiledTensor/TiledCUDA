#pragma once
#include "cell/traits/base.hpp"
#include "config.hpp"

namespace tiledcuda::cell::copy {

namespace traits = tiledcuda::cell::traits;

// FIXME(haruhi): A quick implementation. Improve in the future.
// A basic tile shape is a minimal shape to efficiently use the hardware
// capability.
template <typename Element, typename Base = traits::TraitsBase<Element>>
struct BaseTileShape : public Base {
    static constexpr int elem_per_thread = Base::kNumPerAccess;

    static constexpr int row = 16;
    // This definition aligns with `ldmatrix`.
    // When 32 threads in a warp execute `ldmatrix`, they are arranged in a 2x2
    // thread tile, with each tile containing 8 contiguous threads. Each thread
    // accesses 128 contiguous bits of data in shared memory.
    static constexpr int col = elem_per_thread * 2;
};

enum class CopyInst {
    LoadMat = 0,   // ldmatrix for loading data from shared memory to register.
    StoreMat = 1,  // stmatrix for storing data from register to shared memory.
    LoadS32 = 2,   // ldsm32 for loading 32-bit data from shared memory.
    LoadS128 = 3   // ldsm128 for loading 128-bit data from shared memory.
};

enum class RegLayout {
    TileWMMA = 0,  // Tile layout for TCU WMMA.
};

enum class WarpReuse {
    // TODO(haruhi): It seems that Cir/RowReuseCir/ColReuseCir are not ncessary,
    // thus the reuse mode can be simplified.
    // data are evenly partitioned to be loaded by warps.
    Cont = 0,          // all warps continuously load data, no reuse
    Cir = 1,           // all warps circularly load data, no reuse
    RowReuseCont = 2,  // Row-wise even reuse, warps in the same row
                       // repeatedly load the same data
    RowReuseCir = 3,   // Row-wise circular reuse
    ColReuseCont = 4,  // Column-wise even reuse, warps in the same column
                       // repeatedly load the same data
    ColReuseCir = 5    // Column-wise circular reuse
};

template <typename Element, const int rows, const int warps_per_row,
          const WarpReuse mode>
DEVICE static constexpr int row_exec_count() {
    switch (mode) {
        case WarpReuse::ColReuseCont:
        case WarpReuse::ColReuseCir:
            return rows / BaseTileShape<Element>::row;
        default:  // Cont, Cir, RowReuseCont, RowReuseCir hit this case.
            return rows / BaseTileShape<Element>::row / warps_per_row;
    }
}

template <typename Element, const int cols, const int warps_per_col,
          const WarpReuse mode>
DEVICE static constexpr int col_exec_count() {
    switch (mode) {
        case WarpReuse::RowReuseCont:
        case WarpReuse::RowReuseCir:
            return cols / BaseTileShape<Element>::col;
        default:  // Cont, Cir, ColReuseCont, ColReuseCir hit this case.
            return cols / BaseTileShape<Element>::col / warps_per_col;
    }
}

}  // namespace tiledcuda::cell::copy
