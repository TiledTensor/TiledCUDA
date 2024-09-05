#pragma once

#include "cell/mod.hpp"
#include "types/mod.hpp"

using namespace tiledcuda;
using namespace tiledcuda::cell;
namespace tl = tile_layout;

template <typename InType, typename AccType, typename GemmShape>
struct GemmTraits {
    using BaseShape = traits::BaseTileShape<InType>;
    static constexpr int kChunkK = 32;

    using WarpLayout = tl::RowMajor<2, 2>;
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

    // output C
    using GlobalC = GlobalTile<AccType, tl::RowMajor<kM, kN>>;

    static constexpr int kCMs = kM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kCNs = kN / kWarpPerCol / BaseShape::kTileSize;
    using RegC = RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kCMs, kCNs>>;

    using CStorer = copy::RegToGlobalStorer<GlobalC, RegC, WarpLayout>;
};

template <typename InType, typename AccType, typename IteratorA, typename RegA,
          typename LoaderA, typename IteratorB, typename RegB, typename LoaderB,
          typename GlobalC, typename RegC, typename CStorer>
__global__ void simple_gemm(const InType* dA, const InType* dB, AccType* dC) {
    IteratorA gAs(dA);
    RegA rA;
    LoaderA loader_a;

    IteratorB gBs(dB);
    RegB rB;
    LoaderB loader_b;

    RegC acc;

    for (int k = 0; k < IteratorA::sc1; ++k) {
        loader_a(gAs(k), rA);
        loader_b(gBs(k), rB);
        __syncthreads();

        compute::gemm_(rA, rB, acc);
    }
    __syncthreads();

    GlobalC gC(dC);
    CStorer storer_c;
    storer_c(acc, gC);
}
