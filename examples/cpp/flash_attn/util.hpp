#pragma once

#include "cell/compute/mod.hpp"
#include "cell/mod.hpp"
#include "flash_attn_cpu.hpp"
#include "types/mod.hpp"
#include "util/debug.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace tiledcuda;
using namespace tiledcuda::cell;
using namespace tiledcuda::cell::copy;
namespace tl = tile_layout;

template <const int kM, const int kN, const int kK, const int kP>
using FlashAttentionShape = TileShape<kM, kN, kK, kP>;

float rand_float(float a = 1e-1, float b = 5e-2) {
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

bool check_results(const __half* values1, const __half* values2, int numel) {
    bool passed = true;
    const float epsilon = 1e-1;

    for (int i = 0; i < numel; ++i) {
        if (fabs(__half2float(values1[i]) - __half2float(values2[i])) >
            epsilon) {
            printf("%d-th value differs: %.3f vs. %.3f\n", i,
                   __half2float(values1[i]), __half2float(values2[i]));
            passed = false;
        }
    }
    return passed;
}

template <typename InType, typename AccType, typename WholeShape,
          typename CtaTileShape>
struct FlashAttentionTraits {
    using BaseShape = traits::BaseTileShape<InType>;

    using WarpLayout = tl::RowMajor<4, 1>;

    static constexpr int kWarpPerRow = tl::num_rows<WarpLayout>;
    static constexpr int kWarpPerCol = tl::num_cols<WarpLayout>;
    static_assert(kWarpPerCol == 1, "WarpPerCol must be 1");

    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;

    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;
    static constexpr int kP = dim_size<3, WholeShape>;

    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;
    static constexpr int kTP = dim_size<3, CtaTileShape>;

    // operand A
    using GlobalA = GlobalTile<InType, tl::RowMajor<kTM, kK>>;
    // chunk the K dimension to fit into shared memory
    using GIteratorA = TileIterator<GlobalA, TileShape<kTM, kTK>>;

    using SharedA = SharedTile<InType, tl::RowMajor<kTM, kTK>, true>;

    static constexpr int kAMs = kTM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kAKs = kTK / BaseShape::kTileSize;
    using RegA = RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAMs, kAKs>>;

    using SharedALoader = GlobalToSharedLoader<SharedA, WarpLayout>;
    using RegALoader =
        SharedToRegLoader<RegA, WarpLayout, WarpReuse::kRowReuseCont>;

    // operand B
    using GlobalB = GlobalTile<InType, tl::ColMajor<kK, kN>>;
    using GIteratorB = TileIterator<GlobalB, TileShape<kTK, kTN>>;
    using SharedB = SharedTile<InType, tl::ColMajor<kTK, kTN>, true>;

    static constexpr int kBKs = kTK / BaseShape::kTileSize;
    static constexpr int kBNs = kTN / kWarpPerCol / BaseShape::kTileSize;
    using RegB = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kBKs, kBNs>>;

    using SharedBLoader = GlobalToSharedLoader<SharedB, WarpLayout>;
    using RegBLoader =
        SharedToRegLoader<RegB, WarpLayout, WarpReuse::kColReuseCont>;

    // operand C
    using GlobalC = GlobalTile<InType, tl::ColMajor<kN, kTP>>;
    // chunk the N dimension to fit into shared memory
    using GIteratorC = TileIterator<GlobalC, TileShape<kTN, kTP>>;
    using SharedC = SharedTile<InType, tl::ColMajor<kTN, kTP>, true>;

    static constexpr int kCNs = kTN / BaseShape::kTileSize;
    static constexpr int kCPs = kTP / kWarpPerCol / BaseShape::kTileSize;
    using RegC = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kCNs, kCPs>>;

    using SharedCLoader = GlobalToSharedLoader<SharedC, WarpLayout>;
    using RegCLoader =
        SharedToRegLoader<RegC, WarpLayout, WarpReuse::kColReuseCont>;

    // output D
    // using GlobalD = GlobalTile<AccType, tl::RowMajor<kTM, kTP>>;
    using GlobalD = GlobalTile<InType, tl::RowMajor<kTM, kTP>>;

    static constexpr int kDMs = kTM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kDPs = kTP / kWarpPerCol / BaseShape::kTileSize;
    using RegD = RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kDMs, kDPs>>;
    using RegDCast =
        RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kDMs, kDPs>>;
    using DStorer = copy::RegToGlobalStorer<GlobalD, RegDCast, WarpLayout>;

    static constexpr int kAccMs = kTM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kAccNs = kTN / kWarpPerCol / BaseShape::kTileSize;

    // Reg Acc
    using RegAcc =
        RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kAccMs, kAccNs>>;
    using RegAccCast =
        RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAccMs, kAccNs>>;

    // Convert the accumulator to half
    using ConvertHalf = compute::RegTileConvert<RegAcc, RegAccCast>;
    using ConvertO = compute::RegTileConvert<RegD, RegDCast>;

    using RegVec = RegTile<InType, tl::RowMajor<kAccMs, 2>>;

    using CopyVec = copy::BaseTileCopy<RegVec>;
    using RowMax = compute::MaxReduce<RegAccCast, tl::Layout::kRowMajor>;

    using RowSum = compute::SumReduce<RegAccCast, tl::Layout::kRowMajor>;

    using BroadcastSub =
        compute::BroadcastSub<RegVec, RegAccCast, tl::Layout::kRowMajor>;
    using BroadcastMul =
        compute::BroadcastMul<RegVec, RegDCast, tl::Layout::kRowMajor>;
    using BroadcastDiv =
        compute::BroadcastDiv<RegVec, RegDCast, tl::Layout::kRowMajor>;

    using BlockExp = compute::RegTileExp<RegAccCast>;
    // using BlockAdd = compute::RegTileAdd<RegAccCast>;
    using BlockAdd = compute::RegTileAdd<RegDCast>;

    using VecMax = compute::BaseTileMax<RegVec>;
    using VecAdd = compute::BaseTileAdd<RegVec>;
    using VecSub = compute::BaseTileSub<RegVec>;
    using VecMul = compute::BaseTileMul<RegVec>;
    using VecExp = compute::BaseTileExp<RegVec>;
};
