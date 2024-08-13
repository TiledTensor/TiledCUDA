#pragma once

#include "cell/compute/mod.hpp"
#include "cell/mod.hpp"
#include "types/mod.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace tiledcuda;
using namespace tiledcuda::cell;
namespace tl = tile_layout;

template <const int kM, const int kN, const int kK, const int kP>
using B2BGemmShape = TileShape<kM, kN, kK, kP>;

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

template <typename InType, typename AccType, typename B2BGemmShape>
struct B2BGemmTraits {
    using BaseShape = traits::BaseTileShape<InType>;
    static constexpr int kChunkK = 32;
    // TODO: kChunkN must be match for N-dimension for operand `Acc` and `C`.
    static constexpr int kChunkN = 64;

    using WarpLayout = tl::RowMajor<2, 2>;
    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;
    static constexpr int kWarpPerRow = tl::num_rows<WarpLayout>;
    static constexpr int kWarpPerCol = tl::num_cols<WarpLayout>;

    static constexpr int kM = dim_size<0, B2BGemmShape>;
    static constexpr int kN = dim_size<1, B2BGemmShape>;
    static constexpr int kK = dim_size<2, B2BGemmShape>;
    static constexpr int kP = dim_size<3, B2BGemmShape>;

    // operand A
    using GlobalA = GlobalTile<InType, tl::RowMajor<kM, kK>>;
    using IteratorA = TileIterator<GlobalA, TileShape<kM, kChunkK>>;

    static constexpr int kAMs = kM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kAKs = kChunkK / BaseShape::kTileSize;
    using RegA = RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAMs, kAKs>>;

    using ALoader = copy::GlobalToRegLoader<RegA, WarpLayout,
                                            copy::WarpReuse::kRowReuseCont>;

    // operand B
    using GlobalB = GlobalTile<InType, tl::ColMajor<kK, kN>>;
    using IteratorB = TileIterator<GlobalB, TileShape<kChunkK, kN>>;

    static constexpr int kBKs = kChunkK / BaseShape::kTileSize;
    static constexpr int kBNs = kN / kWarpPerCol / BaseShape::kTileSize;
    using RegB = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kBKs, kBNs>>;

    using BLoader = copy::GlobalToRegLoader<RegB, WarpLayout,
                                            copy::WarpReuse::kColReuseCont>;

    // operand C
    using GlobalC = GlobalTile<InType, tl::ColMajor<kN, kP>>;
    using IteratorC = TileIterator<GlobalC, TileShape<kChunkN, kP>>;

    static constexpr int kCNs = kChunkN / BaseShape::kTileSize;
    static constexpr int kCPs = kP / kWarpPerCol / BaseShape::kTileSize;
    using RegC = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kCNs, kCPs>>;

    using CLoader = copy::GlobalToRegLoader<RegC, WarpLayout,
                                            copy::WarpReuse::kColReuseCont>;

    // output D
    using GlobalD = GlobalTile<AccType, tl::RowMajor<kM, kP>>;

    static constexpr int kDMs = kM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kDPs = kP / kWarpPerCol / BaseShape::kTileSize;
    using RegD = RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kDMs, kDPs>>;
    using DStorer = copy::RegToGlobalStorer<GlobalD, RegD, WarpLayout>;

    static constexpr int kAccMs = kM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kAccNs = kN / kWarpPerCol / BaseShape::kTileSize;

    // Reg Acc
    using RegAcc =
        RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kAccMs, kAccNs>>;
    using RegAccCast =
        RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAccMs, kAccNs>>;

    // Convert the accumulator to half
    using ConvertHalf = compute::RegTileConvertHalf<RegAcc, RegAccCast>;
};
