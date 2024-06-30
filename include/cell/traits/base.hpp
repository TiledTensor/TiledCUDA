#pragma once

#include "types/layout.hpp"

#include <cutlass/numeric_size.h>

namespace tiledcuda::cell::traits {

/// @brief Architecture-specific magic numbers.
/// @tparam Element: the data type of the elements.
template <typename Element>
struct TraitsBase {
    static constexpr int kAccessInBits = 128;  // 128 bits
    static constexpr int kElmentBits = cutlass::sizeof_bits<Element>::value;
    static constexpr int kNumPerAccess = kAccessInBits / kElmentBits;
};

// FIXME(haruhi): This is a quick implementation. Improve it in the future.
// `BasicTileShape` is the minimal shape that efficiently utilizes the
// hardware's capabilities. Strive to organize all the magic numbers around this
// BaseTile more clearly.
template <typename Element, typename Base = TraitsBase<Element>>
struct BaseTileShape : public Base {
    static constexpr int elem_per_thread = Base::kNumPerAccess;

    static constexpr int row = 16;
    // This definition aligns with `ldmatrix`.
    // When 32 threads in a warp execute `ldmatrix`, they are arranged in a 2x2
    // thread tile, with each tile containing 8 contiguous threads. Each thread
    // accesses 128 contiguous bits of data in shared memory.
    static constexpr int col = elem_per_thread * 2;

    // TODO: Comment this;
    static constexpr int sub_row = 2;
    static constexpr int sub_col = 4;
    static constexpr int sub_numel = sub_row * sub_col;
};

using ThreadLdmatrix = tile_layout::ColMajor<16, 2>;
using ThreadWmma = tile_layout::RowMajor<8, 4>;

}  // namespace tiledcuda::cell::traits
