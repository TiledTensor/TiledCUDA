#pragma once

#include <type_traits>

namespace tiledcuda::cell {

template <typename Element>
struct BaseTile {
    using DType = Element;

    static_assert(std::is_same_v<DType, float> ||
                      std::is_same_v<DType, __half> ||
                      std::is_same_v<DType, __bf16>,
                  "Unsupported type.");

    static constexpr int kTileSize = 16;
    static constexpr int kRows = kTileSize;
    static constexpr int kCols = kTileSize;
    static constexpr int kNumel = kRows * kCols;
    static constexpr int kNumelPerThread = kNumel / 32;  // 8

    static constexpr int kRegsPerThread =
        sizeof(DType) * kNumelPerThread / sizeof(uint32);

    DType data[kNumelPerThread];
};

}  // namespace tiledcuda::cell