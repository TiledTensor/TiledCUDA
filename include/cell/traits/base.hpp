#pragma once

#include "types/layout.hpp"

#include <cutlass/numeric_size.h>

namespace tiledcuda::cell::traits {

template <typename Element>
concept BaseType = std::is_same_v<Element, float> ||
    std::is_same_v<Element, __half> || std::is_same_v<Element, cutlass::half_t>;

/// @brief Architecture-specific magic numbers.
/// @tparam Element: the data type of the elements.
template <typename Element>
struct TraitsBase {
    static constexpr int kAccessInBits = 128;  // 128 bits
    static constexpr int kElmentBits = cutlass::sizeof_bits<Element>::value;
    static constexpr int kNumPerAccess = kAccessInBits / kElmentBits;
};

template <typename Element>
requires BaseType<Element>
struct BaseTileShape {
    using DType = Element;

    static constexpr int kTileSize = 16;
    static constexpr int kRows = kTileSize;
    static constexpr int kCols = kTileSize;
    static constexpr int kNumel = kRows * kCols;
    static constexpr int kNumelPerThread = kNumel / 32;           // 8
    static constexpr int kPackedPerThread = kNumelPerThread / 2;  // 4

    // 4 registers used in half / bf16, 8 registers used in float.
    static constexpr int kRegsPerThread =
        sizeof(DType) * kNumelPerThread / sizeof(uint32_t);
};

}  // namespace tiledcuda::cell::traits
