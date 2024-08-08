#include "common/test_utils.hpp"
#include "cuda_utils.hpp"
#include "util/debug.hpp"

#include <cublas_v2.h>

namespace tiledcuda::testing {
using namespace cell;
using namespace cell::copy;
namespace tl = cell::tile_layout;

float rand_float(float a = 1e-3, float b = 1) {
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

bool check_correctness(const float* hc1, const float* hc2, int row, int col) {
    int numel = row * col;
    bool pass_unittest = true;
    static const float eps = 5e-2;

#if defined(DEBUG)
    int cut_off = 128;
    std::stringstream ss;
    ss << std::setprecision(3) << std::endl
       << "ours:" << std::endl
       << 0 << ":\t";
    for (int i = 0; i < cut_off; ++i) {
        ss << hc2[i] << ", ";
        if (i & (i + 1) % 16 == 0) {
            ss << std::endl << (i + 1) / 16 << ":\t";
        }
    }

    ss << std::endl << "cublas:" << std::endl << 0 << ":\t";
    for (int i = 0; i < cut_off; ++i) {
        ss << __half2float(hc1[i]) << ", ";
        if (i & (i + 1) % 16 == 0) {
            ss << std::endl << (i + 1) / 16 << "\t";
        }
    }
    LOG(INFO) << ss.str();
#endif

    double total_diff = 0.;
    double max_abs_diff = FLT_MIN;
    double diff = 0.;

    LOG(INFO) << std::endl;
    for (int i = 0; i < numel; ++i) {
        diff = abs(hc1[i] - hc2[i]);
        max_abs_diff = max_abs_diff < diff ? diff : max_abs_diff;
        total_diff += diff;

        if (diff > eps) {
            LOG(INFO) << i
                      << "-th value has large numeric absolute diff: " << diff
                      << ", Expected: " << __half2float(hc1[i])
                      << "; Got: " << hc2[i] << std::endl;
        }
    }

    double avg_diff = total_diff / numel;
    LOG(INFO) << "Average absolute diff: " << avg_diff
              << ", Max absolute diff: " << max_abs_diff << std::endl;
    if (avg_diff > eps) pass_unittest = false;

    return pass_unittest;
}

// @brief: This implementation interprets A and C as being laid out in row-major
//         order, while B is laid out in column-major order.
//         Matrix A has a row-major layout with dimensions [M, K],
//         Matrix B has a column-major layout with dimensions [K, N],
//         and Matrix C has a row-major layout with dimensions [M, N].
//
//         This is equivalent to the following:
//         Matrix A has a column-major layout with dimensions [K, M],
//         Matrix B has a column-major layout with dimensions [K, N],
//         and Matrix C has a column-major layout with dimensions [N, M].
//         cuBlas is a Fortran-style(column-major) BLAS library,
//         then we compute: C = B^T @ A
//                     [N, M] = [N, K] @ [K, M]
void cublas_sgemm(int m, int n, int k, const float* A, const float* B, float* C,
                  int lda, int ldb, int ldc) {
    float alf = 1.;
    float bet = 0.;

    cublasHandle_t handle;
    CublasCheck(cublasCreate(&handle));
    CublasCheck(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alf, A,
                            lda, B, ldb, &bet, C, ldc));
    CublasCheck(cublasDestroy(handle));
}

// @param strided_k: chunk size to partition the k dimension of the shared
//                   memory tile.
template <typename Element, typename ElementAcc, const int kM, const int kN,
          const int kK, typename WarpLayout_, const int kChunkK>
struct TestTraits {
    using BaseShape = traits::BaseTileShape<Element>;

    /// ======== 1. configurate threads and warp layout in a CTA ============
    using WarpLayout = WarpLayout_;
    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;
    static constexpr int kWarpPerRow = tl::num_rows<WarpLayout>;
    static constexpr int kWarpPerCol = tl::num_cols<WarpLayout>;

    using GlobalA = GlobalTile<Element, tl::RowMajor<kM, kK>>;

    /// == 2. configurate tile transfer between global and shared using CuTe ==
    using Layout1 = tl::RowMajor<kM, kK>;
    using SharedA = SharedTile<Element, Layout1>;
    using LoadSharedA = GlobalToSharedLoader<SharedA, WarpLayout>;
    using TileIteratorA = TileIterator<SharedA, TileShape<kM, kChunkK>>;

    // for testing swizzled layout
    using Layout2 = tl::Swizzled<tl::RowMajor<kM, kK>, 2, 3, 3>;
    using SharedA2 = SharedTile<Element, Layout2>;
    using LoadSharedA2 = GlobalToSharedLoader<SharedA2, WarpLayout>;
    using TileIteratorA2 = TileIterator<SharedA2, TileShape<kM, kChunkK>>;

    using GlobalB = GlobalTile<Element, tl::RowMajor<kN, kK>>;
    using SharedB = SharedTile<Element, tl::RowMajor<kN, kK>>;
    using LoadSharedB = GlobalToSharedLoader<SharedB, WarpLayout>;

    /// === 3. configurate tile transfer between shared and register loader ===
    // shared tile for operand A

    // shared tile for operand B
    using SharedB2 = SharedTile<Element, tl::ColMajor<kK, kN>>;
    using TileIteratorB = TileIterator<SharedB2, TileShape<kChunkK, kN>>;

    static_assert(TileIteratorA::sc1 == TileIteratorB::sc0,
                  "mismatched K dimension!");

    // register tile for operand A, calculate register usage for operand A
    // warp tile shape for the operand A
    static constexpr int kAMs = kM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kAKs = kChunkK / BaseShape::kTileSize;
    using RegA = RegTile<BaseTileRowMajor<Element>, tl::RowMajor<kAMs, kAKs>>;
    // load RegTileA from shared
    using LoadRegA =
        SharedToRegLoader<RegA, WarpLayout, WarpReuse::kRowReuseCont>;

    // register tile for operand B, calculate register usage for operand B
    static constexpr int kBKs = kChunkK / BaseShape::kTileSize;
    static constexpr int kBNs = kN / kWarpPerCol / BaseShape::kTileSize;
    using RegB = RegTile<BaseTileColMajor<Element>, tl::ColMajor<kBKs, kBNs>>;
    // load RegTileB from shared
    using LoadRegB =
        SharedToRegLoader<RegB, WarpLayout, WarpReuse::kColReuseCont>;

    // shared tile for output C
    using SharedC = SharedTile<ElementAcc, tl::RowMajor<kM, kN>>;
    using GlobalC = GlobalTile<ElementAcc, tl::RowMajor<kM, kN>>;
    using StoreSharedC = SharedToGlobalStorer<SharedC, WarpLayout>;

    // register tile for output C
    // calculate register usage for output C
    static constexpr int kCMs = kM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kCNs = kN / kWarpPerCol / BaseShape::kTileSize;
    using RegC =
        RegTile<BaseTileRowMajor<ElementAcc>, tl::RowMajor<kCMs, kCNs>>;

    // store RegTileC to shared
    using StoreRegC = RegToSharedStorer<RegC, WarpLayout>;
};

}  // namespace tiledcuda::testing
