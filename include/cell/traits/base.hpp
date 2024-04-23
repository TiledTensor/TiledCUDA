#pragma once
#include <cutlass/numeric_size.h>

namespace tiledcuda::cell::traits {

// FIXME(ying). The swizzle function requires a data tile with a minimal
// shape of <8, 32> for the <2, 3, 3> case, and a minimal shape of <8, 64> for
// the <3, 3, 3> case. Here requires some check to ensure that the data tile
// meets these requirements before using this function.
template <const int N>
static constexpr int kSwizzle = (N == 32 ? 2 : 3);

template <typename Element>
struct TraitsBase {
    static constexpr int kAccessInBits = 128;  // 128 bits
    static constexpr int kElmentBits = cutlass::sizeof_bits<Element>::value;
    static constexpr int kNumPerAccess = kAccessInBits / kElmentBits;
};
}  // namespace tiledcuda::cell::traits
