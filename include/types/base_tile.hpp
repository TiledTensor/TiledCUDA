#pragma once

namespace tiledcuda::cell {

template <typename Element>
struct BaseTile {
    using DType = Element;

    static_assert(std::is_same_v<DType, float> ||
                      std::is_same_v<DType, cutlass::half_t> ||
                      std::is_same_v<DType, __half>,
                  "Unsupported type.");

    static constexpr int kTileSize = 16;
    static constexpr int kRows = kTileSize;
    static constexpr int kCols = kTileSize;
    static constexpr int kNumel = kRows * kCols;
    static constexpr int kNumelPerThread = kNumel / 32;           // 8
    static constexpr int kPackedPerThread = kNumelPerThread / 2;  // 4

    // 4 registers used in half/bf16, 8 registers used in float.
    static constexpr int kRegsPerThread =
        sizeof(DType) * kNumelPerThread / sizeof(uint32_t);

    // for write access
    DEVICE DType& operator()(int x) { return data_[x]; }

    // for read access
    DEVICE const DType& operator()(int x) const { return data_[x]; }

    DType data_[kNumelPerThread];
};

}  // namespace tiledcuda::cell
