#pragma once

#include "cell/compute/mod.hpp"
#include "cell/mod.hpp"
#include "types/mod.hpp"

#include <cublas_v2.h>
#include <float.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace tiledcuda;
using namespace tiledcuda::cell;
using namespace tiledcuda::cell::copy;
namespace tl = tile_layout;

template <const int kM, const int kN, const int kK, const int kP>
using B2BGemmShape = TileShape<kM, kN, kK, kP>;

float rand_float(float a = 1e-1, float b = 5e-2) {
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

/* In this implementation, A and D are interpreted as being laid out in
   row-major, and B, C is interpreted as being laid out in column-major.

  A and D are laid out in row-major fashion
  B and C are laid out in column-major fashion

  acc[m, n] = A[m, k] @ B[k, n]
    D[m, p] = acc[m, n] @ C[n, p]
*/
void cublas_two_gemms(int kM, int kN, int kK, int kP, int kBatch,
                      const __half* As, const __half* Bs, const __half* Cs,
                      __half* Ds, __half* accs) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    __half alf = static_cast<__half>(1.);
    __half bet = static_cast<__half>(0.);

    const __half* A = As;
    const __half* B = Bs;
    const __half* C = Cs;
    __half* acc = accs;
    __half* D = Ds;

    for (int b = 0; b < kBatch; ++b) {
        A += b * kM * kK;
        B += b * kK * kN;
        C += b * kM * kN;
        acc += b * kM * kN;
        D += b * kM * kP;

        // acc   = A @ B
        // acc^T = B^T @ A^T
        // [n, m] = [n, k] @ [k, m]
        cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N /* transb*/, kN, kM, kK,
                    &alf, B, kK, A, kK, &bet, acc, kN);

        // D and acc are laid out in row-major fashion, while C is in column
        // major fashion. Operands of cuBLAS is by default in column fashion.
        // D = acc @ C
        // D^T = C^T @ acc^T; [p, m] = [p, n] @ [n, m]
        cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N /* transb*/, kP, kM, kN,
                    &alf, C, kN, acc, kN, &bet, D, kP);
    }

    cublasDestroy(handle);
}

bool check_results(const float* values1, const __half* values2, int numel,
                   float epsilon) {
    bool passed = true;

    float v2 = 0.;

    double total_diff = 0.;
    double max_abs_diff = FLT_MIN;
    double diff = 0.;

    for (int i = 0; i < numel; ++i) {
        v2 = __half2float(values2[i]);
        diff = abs(values1[i] - v2);
        max_abs_diff = max_abs_diff < diff ? diff : max_abs_diff;
        total_diff += diff;

#ifdef DEBUG
        if (diff > epsilon) {
            printf("%d-th value has large differences: %.3f vs. %.3f\n", i,
                   values1[i], v2);
        }
#endif
    }

    double avg_diff = total_diff / numel;
    if (avg_diff > epsilon) passed = false;

    return passed;
}

template <typename InType, typename AccType, typename WholeShape,
          typename CtaTileShape, typename WarpLayout>
struct B2BGemmTraits {
    using BaseShape = traits::BaseTileShape<InType>;

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

    static const bool kUseSwizzling = true;

    using SharedA = SharedTile<InType, tl::RowMajor<kTM, kTK>, kUseSwizzling>;

    static constexpr int kAMs = kTM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kAKs = kTK / BaseShape::kTileSize;
    using RegA = RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAMs, kAKs>>;

    using SharedALoader = GlobalToSharedLoader<SharedA, WarpLayout>;
    using RegALoader =
        SharedToRegLoader<RegA, WarpLayout, WarpReuse::kRowReuseCont>;

    // operand B
    using GlobalB = GlobalTile<InType, tl::ColMajor<kK, kN>>;
    using GIteratorB = TileIterator<GlobalB, TileShape<kTK, kTN>>;
    using SharedB = SharedTile<InType, tl::ColMajor<kTK, kTN>, kUseSwizzling>;

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
    using SharedC = SharedTile<InType, tl::ColMajor<kTN, kTP>, kUseSwizzling>;

    static constexpr int kCNs = kTN / BaseShape::kTileSize;
    static constexpr int kCPs = kTP / kWarpPerCol / BaseShape::kTileSize;
    using RegC = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kCNs, kCPs>>;

    using SharedCLoader = GlobalToSharedLoader<SharedC, WarpLayout>;
    using RegCLoader =
        SharedToRegLoader<RegC, WarpLayout, WarpReuse::kColReuseCont>;

    // output D
    using GlobalD = GlobalTile<AccType, tl::RowMajor<kTM, kTP>>;

    static constexpr int kDMs = kTM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kDPs = kTP / kWarpPerCol / BaseShape::kTileSize;
    using RegD = RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kDMs, kDPs>>;
    using DStorer = copy::RegToGlobalStorer<GlobalD, RegD, WarpLayout>;

    static constexpr int kAccMs = kTM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kAccNs = kTN / kWarpPerCol / BaseShape::kTileSize;

    // Reg Acc
    using RegAcc =
        RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kAccMs, kAccNs>>;
    using RegAccCast =
        RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAccMs, kAccNs>>;

    // Convert the accumulator to half
    using ConvertHalf = compute::RegTileConvert<RegAcc, RegAccCast>;
};
