#pragma once

#include "cell/mod.hpp"
#include "types/mod.hpp"

using namespace tiledcuda;
using namespace tiledcuda::cell;
using namespace tiledcuda::cell::copy;

namespace tl = tile_layout;

template <const int kM, const int kN, const int kK>
using GemmShape = TileShape<kM, kN, kK>;

template <typename InType, typename AccType, typename WholeShape,
          typename CtaTileShape, const int kRK, typename WarpLayout>
struct KeGemmTraits {
    using BaseShape = traits::BaseTileShape<InType>;

    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;
    static constexpr int kWarpPerRow = tl::num_rows<WarpLayout>;
    static constexpr int kWarpPerCol = tl::num_cols<WarpLayout>;

    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;

    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;

    static const bool kSwizzled = true;

    // Total data access for operand A in global memory
    using GlobalA = GlobalTile<InType, tl::RowMajor<kTM, kK, kK>>;
    // Access a single global tile for operand A
    using GIteratorA = TileIterator<GlobalA, TileShape<kTM, kTK>>;

    // Shared Tile for operand A
    using SharedA = SharedTile<InType, tl::RowMajor<kTM, kTK>, kSwizzled>;
    // Access a single register tile for operand A
    using SIteratorA = TileIterator<SharedA, TileShape<kTM, kRK>>;

    // Register tile for a single thread of operand A
    static constexpr int kAMs = kTM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kAKs = kRK / BaseShape::kTileSize;
    using RegA = RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAMs, kAKs>>;

    // Loaders for operand A
    using G2SLoaderA = GlobalToSharedLoader<SharedA, WarpLayout>;
    using S2RLoaderA =
        SharedToRegLoader<RegA, WarpLayout, WarpReuse::kRowReuseCont>;

    // Total data access for operand B in global memory
    using GlobalB = GlobalTile<InType, tl::ColMajor<kK, kTN, kK>>;
    // Access a single global tile for operand B
    using GIteratorB = TileIterator<GlobalB, TileShape<kTK, kTN>>;

    // Shared Tile for operand B
    using SharedB = SharedTile<InType, tl::ColMajor<kTK, kTN>, kSwizzled>;
    // Access a single register tile for operand B
    using SIteratorB = TileIterator<SharedB, TileShape<kRK, kTN>>;

    static_assert(GIteratorA::sc1 == GIteratorB::sc0,
                  "mismatched K dimension!");
    static_assert(SIteratorA::sc1 == SIteratorB::sc0,
                  "mismatched K dimension!");

    // Register tile for a single thread of operand A
    static constexpr int kBKs = kRK / BaseShape::kTileSize;
    static constexpr int kBNs = kTN / kWarpPerCol / BaseShape::kTileSize;
    using RegB = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kBKs, kBNs>>;

    using G2SLoaderB = GlobalToSharedLoader<SharedB, WarpLayout>;
    using S2RLoaderB =
        SharedToRegLoader<RegB, WarpLayout, WarpReuse::kColReuseCont>;

    // Global Tile for output C
    using GlobalC = GlobalTile<InType, tl::RowMajor<kTM, kTN, kN>>;
    // Shared Tile for output C
    using SharedC = SharedTile<InType, tl::RowMajor<kTM, kTN>, false>;

    // Register Tile for output C
    static constexpr int kCMs = kTM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kCNs = kTN / kWarpPerCol / BaseShape::kTileSize;
    using RegC = RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kCMs, kCNs>>;

    using RegCHalf =
        RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kCMs, kCNs>>;
    using ConvertHalf = compute::RegTileConvert<RegC, RegCHalf>;

    using R2SStorerC = RegToSharedStorer<RegCHalf, WarpLayout>;
    using S2GStorerC = SharedToGlobalStorer<SharedC, WarpLayout>;
};

template <typename InType, typename AccType,                  //
          const int kM, const int kN, const int kK,           //
          const int kTM, const int kTN, const int kTK,        //
          typename GIteratorA, typename SIteratorA,           //
          typename SharedA, typename RegA,                    //
          typename G2SLoaderA, typename S2RLoaderA,           //
          typename GIteratorB, typename SIteratorB,           //
          typename SharedB, typename RegB,                    //
          typename G2SLoaderB, typename S2RLoaderB,           //
          typename GlobalC, typename SharedC, typename RegC,  //
          typename RegCHalf, typename ConvertHalf,            //
          typename R2SStorerC, typename S2GStorerC>
__global__ void gemm(const InType* dA, const InType* dB, InType* dC) {
    int offset_a = blockIdx.x * kTM * kK;
    int offset_b = blockIdx.y * kTN * kK;
    int offset_c = blockIdx.x * kTM * kN + blockIdx.y * kTN;

    extern __shared__ __align__(sizeof(double)) unsigned char buf[];
    InType* sA_ptr = reinterpret_cast<InType*>(buf);
    InType* sB_ptr = sA_ptr + SIteratorA::Tile::kNumel;
    InType* sC_ptr = reinterpret_cast<InType*>(buf);

    // declare tiles, iterators and loaders
    GIteratorA gAs(dA + offset_a);
    SIteratorA sAs(sA_ptr);

    GIteratorB gBs(dB + offset_b);
    SIteratorB sBs(sB_ptr);

    SharedA sA(sA_ptr);
    RegA rA;

    SharedB sB(sB_ptr);
    RegB rB;

    RegC acc;
    RegCHalf acc_half;
    SharedC sC(sC_ptr);
    GlobalC gC(dC + offset_c);

    G2SLoaderA g2s_a;
    S2RLoaderA s2r_a;

    G2SLoaderB g2s_b;
    S2RLoaderB s2r_b;

    R2SStorerC r2s_c;
    S2GStorerC s2g_c;

    ConvertHalf cast;

    for (int k1 = 0; k1 < GIteratorA::sc1; ++k1) {
        g2s_a(gAs(k1), sA);
        g2s_b(gBs(k1), sB);
        __copy_async();
        __syncthreads();

        for (int k2 = 0; k2 < SIteratorA::sc1; ++k2) {
            s2r_a(sAs(k2), rA);
            s2r_b(sBs(k2), rB);

            compute::gemm_(rA, rB, acc);
        }
    }
    cast(acc, acc_half);
    r2s_c(acc_half, sC);
    __syncthreads();

    s2g_c(sC, gC);
}
