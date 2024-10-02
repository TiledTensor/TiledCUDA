#pragma once

#include "cell/mod.hpp"
#include "types/mod.hpp"

using namespace tiledcuda;
using namespace tiledcuda::cell;
namespace tl = tile_layout;

template <const int kM, const int kN, const int kK>
using GemmShape = TileShape<kM, kN, kK>;

template <typename InType, typename AccType, typename WholeShape, const int kTM,
          const int kTN, typename WarpLayout>
struct GemmTraits {
    using BaseShape = traits::BaseTileShape<InType>;
    static constexpr int kChunkK = 64;

    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;
    static constexpr int kWarpPerRow = tl::num_rows<WarpLayout>;
    static constexpr int kWarpPerCol = tl::num_cols<WarpLayout>;

    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;

    // operand A
    using GlobalA = GlobalTile<InType, tl::RowMajor<kTM, kK, kK>>;
    using IteratorA = TileIterator<GlobalA, TileShape<kTM, kChunkK>>;

    static constexpr int kAMs = kTM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kAKs = kChunkK / BaseShape::kTileSize;
    using RegA = RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAMs, kAKs>>;

    using ALoader = copy::GlobalToRegLoader<RegA, WarpLayout,
                                            copy::WarpReuse::kRowReuseCont>;

    // operand B
    using GlobalB = GlobalTile<InType, tl::ColMajor<kK, kTN, kK>>;
    using IteratorB = TileIterator<GlobalB, TileShape<kChunkK, kTN>>;

    static constexpr int kBKs = kChunkK / BaseShape::kTileSize;
    static constexpr int kBNs = kTN / kWarpPerCol / BaseShape::kTileSize;
    using RegB = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kBKs, kBNs>>;

    using BLoader = copy::GlobalToRegLoader<RegB, WarpLayout,
                                            copy::WarpReuse::kColReuseCont>;

    // output C
    using GlobalC = GlobalTile<AccType, tl::RowMajor<kTM, kTN, kN>>;

    static constexpr int kCMs = kTM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kCNs = kTN / kWarpPerCol / BaseShape::kTileSize;
    using RegC = RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kCMs, kCNs>>;

    using CStorer = copy::RegToGlobalStorer<GlobalC, RegC, WarpLayout>;
};

template <typename InType, typename AccType,                    //
          const int kM, const int kN, const int kK,             //
          const int kTM, const int kTN,                         //
          typename IteratorA, typename RegA, typename ALoader,  //
          typename IteratorB, typename RegB, typename BLoader,  //
          typename GlobalC, typename RegC, typename CStorer>
__global__ void simple_gemm(const InType* dA, const InType* dB, AccType* dC) {
    int offset_a = blockIdx.x * kTM * kK;
    int offset_b = blockIdx.y * kTN * kK;
    int offset_c = blockIdx.x * kTM * kN + blockIdx.y * kTN;

    IteratorA gAs(dA + offset_a);
    RegA rA;
    ALoader loader_a;

    IteratorB gBs(dB + offset_b);
    RegB rB;
    BLoader loader_b;

    RegC acc;
    GlobalC gC(dC + offset_c);
    CStorer storer_c;

    for (int k = 0; k < IteratorA::sc1; ++k) {
        loader_a(gAs(k), rA);
        loader_b(gBs(k), rB);
        __syncthreads();

        compute::gemm_(rA, rB, acc);
    }
    __syncthreads();

    storer_c(acc, gC);
}
