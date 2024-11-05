types = """#include "cell/mod.hpp"
#include "types/mod.hpp"
#include "kernel.h"

using InType = __half;
using AccType = float;

static constexpr int kM = {kM};
static constexpr int kN = {kN};
static constexpr int kK = {kK};

static constexpr int kTM = {kTM};
static constexpr int kTN = {kTN};

using WarpLayout = tl::RowMajor<{warp_per_row}, {warp_per_col}>;

using BaseShape = traits::BaseTileShape<InType>;
static constexpr int kChunkK = {kChunkK};

static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;
static constexpr int kWarpPerRow = tl::num_rows<WarpLayout>;
static constexpr int kWarpPerCol = tl::num_cols<WarpLayout>;

using GlobalA = GlobalTile<InType, tl::RowMajor<kTM, kK, kK>>;
using IteratorA = GTileIterator<GlobalA, TileShape<kTM, kChunkK>>;

static constexpr int kAMs = kTM / kWarpPerRow / BaseShape::kTileSize;
static constexpr int kAKs = kChunkK / BaseShape::kTileSize;
using RegA = RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAMs, kAKs>>;

using ALoader = copy::GlobalToRegLoader<RegA, WarpLayout,
                                        copy::WarpReuse::kRowReuseCont>;

using GlobalB = GlobalTile<InType, tl::ColMajor<kK, kTN, kK>>;
using IteratorB = GTileIterator<GlobalB, TileShape<kChunkK, kTN>>;

static constexpr int kBKs = kChunkK / BaseShape::kTileSize;
static constexpr int kBNs = kTN / kWarpPerCol / BaseShape::kTileSize;
using RegB = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kBKs, kBNs>>;

using BLoader = copy::GlobalToRegLoader<RegB, WarpLayout,
                                        copy::WarpReuse::kColReuseCont>;

using GlobalC = GlobalTile<AccType, tl::RowMajor<kTM, kTN, kN>>;

static constexpr int kCMs = kTM / kWarpPerRow / BaseShape::kTileSize;
static constexpr int kCNs = kTN / kWarpPerCol / BaseShape::kTileSize;
using RegC = RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kCMs, kCNs>>;

using CStorer = copy::RegToGlobalStorer<GlobalC, RegC, WarpLayout>;

int block_x = CeilDiv<kM, kTM>;
int block_y = CeilDiv<kN, kTN>;
"""

entry = """
extern "C" int kernel_entry(__half* parameter1, __half* parameter2,
                            float* paramter3) {
    auto kernel =
        &gemm<InType, AccType, kM, kN, kK, kTM, kTN, IteratorA, RegA, ALoader,
              IteratorB, RegB, BLoader, GlobalC, RegC, CStorer>;

    kernel<<<dim3(block_x, block_y, 1), dim3(kThreads, 1, 1)>>>(
        parameter1, parameter2, paramter3);

    return 0;
}
"""
