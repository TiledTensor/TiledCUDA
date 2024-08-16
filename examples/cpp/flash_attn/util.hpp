#pragma once

#include "cell/compute/mod.hpp"
#include "cell/mod.hpp"
#include "types/mod.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace tiledcuda;
using namespace tiledcuda::cell;
namespace tl = tile_layout;

template <const int kN, const int kD>
using FlashAttnShape = TileShape<kN, kD>;

float rand_float(float a = 1e-3, float b = 1) {
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

// In this implementation, A and D are interpreted as being laid out in
// row-major, and B, C is interpreted as being laid out in column-major.
void naive_back2back_gemm(int kM, int kN, int kK, int kP, const __half* A,
                          const __half* B, const __half* C, float* D,
                          __half* acc) {
    for (int i = 0; i < kM; ++i) {
        for (int j = 0; j < kN; ++j) {
            __half s = 0.;
            for (int k = 0; k < kK; ++k) {
                s += A[i * kK + k] * B[k + kK * j];
            }
            acc[i * kN + j] = s;
        }
    }

    for (int i = 0; i < kM; ++i) {
        for (int j = 0; j < kP; ++j) {
            float s = 0.;
            for (int k = 0; k < kN; ++k) {
                s +=
                    __half2float(acc[i * kN + k]) * __half2float(C[k + kN * j]);
            }
            D[i * kP + j] = s;
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

template <typename InType, typename OutType, typename FlashAttnShape>
struct FlashAttnTraits {
    using BaseShape = traits::BaseTileShape<InType>;

    static constexpr int Bc = 32;
    static constexpr int Br = 32;

    // Q, K, V, and O are all of shape [N, D]
    static constexpr int kN = dim_size<0, FlashAttnShape>;
    static constexpr int kD = dim_size<1, FlashAttnShape>;

    using WarpLayout = tl::RowMajor<2, 2>;
    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;
    static constexpr int kWarpPerRow = tl::num_rows<WarpLayout>;
    static constexpr int kWarpPerCol = tl::num_cols<WarpLayout>;

    // operand Q
    using GlobalQ = GlobalTile<InType, tl::RowMajor<Br, kD>>;

    // Don't need to split Q.
    static constexpr int kQRows = Br / BaseShape::kTileSize;
    static constexpr int kQCols = kD / BaseShape::kTileSize;

    using RegQ =
        RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kQRows, kQCols>>;
    using QLoader =
        copy::GlobalToRegLoader<RegQ, WarpLayout, copy::WarpReuse::kCont>;

    // operand K
    using GlobalK = GlobalTile<InType, tl::ColMajor<kD, kN>>;
    using IteratorK = TileIterator<GlobalK, TileShape<kD, Bc>>;
    static constexpr int kKRows = kD / BaseShape::kTileSize;
    static constexpr int kKCols = kN / kWarpPerCol / BaseShape::kTileSize;

    using RegK =
        RegTile<BaseTileColMajor<InType>, tl::ColMajor<kKRows, kKCols>>;
    using KLoader = copy::GlobalToRegLoader<RegK, WarpLayout,
                                            copy::WarpReuse::kRowReuseCont>;
    // operand V
    using GlobalV = GlobalTile<InType, tl::ColMajor<kN, kD>>;
    using IteratorV = TileIterator<GlobalV, TileShape<Bc, kD>>;
    static constexpr int kVRows = Bc / BaseShape::kTileSize;
    static constexpr int kVCols = kD / kWarpPerCol / BaseShape::kTileSize;

    using RegV =
        RegTile<BaseTileColMajor<InType>, tl::ColMajor<kVRows, kVCols>>;
    using VLoader = copy::GlobalToRegLoader<RegV, WarpLayout,
                                            copy::WarpReuse::kRowReuseCont>;

    // output O
    using GlobalO = GlobalTile<OutType, tl::RowMajor<Br, kD>>;

    static constexpr int kORows = Br / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kOCols = kD / kWarpPerCol / BaseShape::kTileSize;

    using RegO =
        RegTile<BaseTileRowMajor<OutType>, tl::RowMajor<kORows, kOCols>>;
    using OStorer = copy::RegToGlobalStorer<GlobalO, RegO, WarpLayout>;

    // Accumulator
    static constexpr int kAccRows = Br / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kAccCols = Bc / kWarpPerCol / BaseShape::kTileSize;

    using RegAcc =
        RegTile<BaseTileRowMajor<OutType>, tl::RowMajor<kAccRows, kAccCols>>;
    using RegAccCast =
        RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAccRows, kAccCols>>;

    using Convert = compute::RegTileConvert<RegAcc, RegAccCast>;

    static constexpr int kReduceHeight =
        Br / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kReduceWidth = 2;

    using RegVec = RegTile<InType, tl::RowMajor<kReduceHeight, kReduceWidth>>;

    using CopyVec = copy::BaseTileCopy<RegVec>;

    // using ReduceMax = compute::ReduceMax<RegAccCast, tl::Layout::kRowMajor>;
};
