#pragma once

#include <cutlass/numeric_size.h>

namespace tiledcuda::cell::traits {

/// @brief Architecture-specific magic numbers.
/// @tparam Element
template <typename Element>
struct TraitsBase {
    static constexpr int kAccessInBits = 128;  // 128 bits
    static constexpr int kElmentBits = cutlass::sizeof_bits<Element>::value;
    static constexpr int kNumPerAccess = kAccessInBits / kElmentBits;
};
}  // namespace tiledcuda::cell::traits
