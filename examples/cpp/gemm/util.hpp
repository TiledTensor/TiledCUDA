#pragma once

#include "cell/mod.hpp"
#include "types/mod.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace tiledcuda;
using namespace tiledcuda::cell;
namespace tl = tile_layout;

template <const int kM, const int kN, const int kK>
using GemmShape = TileShape<kM, kN, kK>;

float rand_float(float a = 1e-3, float b = 1) {
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

// In this implementation, A and C are interpreted as being laid out in
// row-major, and B is interpreted as being laid out in column-major.
void naive_gemm(int kM, int kN, int kK,  //
                const __half* A, const __half* B, float* C) {
    for (int i = 0; i < kM; ++i) {
        for (int j = 0; j < kN; ++j) {
            float s = 0.;
            for (int k = 0; k < kK; ++k) {
                s += __half2float(A[i * kK + k]) * __half2float(B[k + kK * j]);
            }
            C[i * kN + j] = s;
        }
    }
}

bool check_results(const float* values1, const float* values2, int numel) {
    bool passed = true;
    const float epsilon = 1e-3;

    for (int i = 0; i < numel; ++i) {
        if (fabs(values1[i] - values2[i]) > epsilon) {
            printf("Diff: %.2f vs. %.2f\n", values1[i], values2[i]);
            passed = false;
            break;
        }
    }
    return passed;
}

template <typename InType, typename AccType, typename GemmShape>
struct GemmTraits {
    using BaseShape = traits::BaseTileShape<InType>;
    static constexpr int kChunkK = 16;

    using WarpLayout = tl::RowMajor<1, 1>;
    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;
    static constexpr int kWarpPerRow = tl::num_rows<WarpLayout>;
    static constexpr int kWarpPerCol = tl::num_cols<WarpLayout>;

    static constexpr int kM = dim_size<0, GemmShape>;
    static constexpr int kN = dim_size<1, GemmShape>;
    static constexpr int kK = dim_size<2, GemmShape>;

    // operand A
    using GlobalA = GlobalTile<InType, tl::RowMajor<kM, kK>>;
    using IteratorA = TileIterator<GlobalA, TileShape<kM, kChunkK>>;

    static constexpr int kAMs = kM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kAKs = kChunkK / BaseShape::kTileSize;
    using RegA = RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAMs, kAKs>>;

    using ALoader = copy::GlobalToRegLoader<GlobalA, RegA, WarpLayout,
                                            copy::WarpReuse::kRowReuseCont>;

    // operand B
    using GlobalB = GlobalTile<InType, tl::ColMajor<kK, kN>>;
    using IteratorB = TileIterator<GlobalB, TileShape<kChunkK, kN>>;

    static constexpr int kBKs = kChunkK / BaseShape::kTileSize;
    static constexpr int kBNs = kN / kWarpPerCol / BaseShape::kTileSize;
    using RegB = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kBKs, kBNs>>;

    using BLoader = copy::GlobalToRegLoader<GlobalB, RegB, WarpLayout,
                                            copy::WarpReuse::kColReuseCont>;

    // output C
    using GlobalC = GlobalTile<AccType, tl::RowMajor<kM, kN>>;

    static constexpr int kCMs = kM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kCNs = kN / kWarpPerCol / BaseShape::kTileSize;
    using RegC = RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kCMs, kCNs>>;

    using CStorer = copy::RegToGlobalStorer<GlobalC, RegC, WarpLayout>;
};
