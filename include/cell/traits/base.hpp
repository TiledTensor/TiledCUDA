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

// FIXME(haruhi): This is a quick implementation. These magic numbers SHOULD be
// carefully checked to ensure the correctness. `BasicTileShape` is the minimal
// shape that efficiently utilizes the hardware's capabilities. Strive to
// organize all the magic numbers around this BaseTile more clearly.
template <typename Element, typename Base = TraitsBase<Element>>
struct BaseTileShape : public Base {
    static constexpr int elem_per_thread = Base::kNumPerAccess;

    static constexpr int row = 16;
    // This definition aligns with `ldmatrix`.
    // When 32 threads in a warp execute `ldmatrix`, they are arranged in a 2x2
    // thread tile, with each tile containing 8 contiguous threads. Each thread
    // accesses 128 contiguous bits of data in shared memory.

    // FIXME(haruhi): This is a risky assumption and leads to very dangerous
    // implementations.
    // It defines the `BasicTile` shape for ldmatrix. In BaseTileShape for wmma,
    // the column count is always 16, irrespective of the data type. Resolve
    // this inconsistency.
    static constexpr int col = elem_per_thread * 2;

    // A 16 x 128-bit BaseShape has a sub-structure.
    // Within a 16 x 128-bit BaseShape, each thread processes 4 fragments in
    // this BaseTile. For example, if the input type is half, each access
    // handles 8 halves (128 / 16). Therefore, each fragment has a row-major
    // layout with a shape of [2, 8 / (4 / 2)] = [2, 4].
    // NOTE: this check assumes that the fragment is ALWAYS row-major.
    static constexpr int sub_row = 2;
    static constexpr int sub_col = elem_per_thread / 2;
    static constexpr int sub_numel = sub_row * sub_col;
};

using ThreadLdmatrix = tile_layout::ColMajor<16, 2>;
using ThreadWmma = tile_layout::RowMajor<8, 4>;

}  // namespace tiledcuda::cell::traits
