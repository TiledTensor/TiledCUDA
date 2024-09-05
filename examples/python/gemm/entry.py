types = """#pragma once

#include "cell/mod.hpp"
#include "types/mod.hpp"
#include "kernel.h"

using InType = __half;
using AccType = float;

static constexpr int kM = {kM};
static constexpr int kN = {kN};
static constexpr int kK = {kK};

using BaseShape = traits::BaseTileShape<InType>;
static constexpr int kChunkK = 32;

using WarpLayout = tl::RowMajor<2, 2>;
static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;

static constexpr int kWarpPerRow = tl::num_rows<WarpLayout>;
static constexpr int kWarpPerCol = tl::num_cols<WarpLayout>;

// operand A
using GlobalA = GlobalTile<InType, tl::RowMajor<kM, kK>>;
using IteratorA = TileIterator<GlobalA, TileShape<kM, kChunkK>>;

static constexpr int kAMs = kM / kWarpPerRow / BaseShape::kTileSize;
static constexpr int kAKs = kChunkK / BaseShape::kTileSize;
using RegA = RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAMs, kAKs>>;

using ALoader =
    copy::GlobalToRegLoader<RegA, WarpLayout, copy::WarpReuse::kRowReuseCont>;

// operand B
using GlobalB = GlobalTile<InType, tl::ColMajor<kK, kN>>;
using IteratorB = TileIterator<GlobalB, TileShape<kChunkK, kN>>;

static constexpr int kBKs = kChunkK / BaseShape::kTileSize;
static constexpr int kBNs = kN / kWarpPerCol / BaseShape::kTileSize;
using RegB = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kBKs, kBNs>>;

using BLoader =
    copy::GlobalToRegLoader<RegB, WarpLayout, copy::WarpReuse::kColReuseCont>;

// output C
using GlobalC = GlobalTile<AccType, tl::RowMajor<kM, kN>>;

static constexpr int kCMs = kM / kWarpPerRow / BaseShape::kTileSize;
static constexpr int kCNs = kN / kWarpPerCol / BaseShape::kTileSize;
using RegC = RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kCMs, kCNs>>;

using CStorer = copy::RegToGlobalStorer<GlobalC, RegC, WarpLayout>;
"""

entry = """
extern "C" int kernel_entry(__half * parameter1, __half * parameter2,
                            float * paramter3) {
    auto kernel = &gemm < InType, AccType, IteratorA, RegA, ALoader, IteratorB,
                        RegB, BLoader, GlobalC, RegC, CStorer > ;

    kernel << <dim3(1, 1, 1), dim3(kThreads, 1, 1) >> >(parameter1, parameter2,
                                                    paramter3);
    return 0;
}
"""
