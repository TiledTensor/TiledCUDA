#include "cell/mod.hpp"
#include "common/test_utils.hpp"
#include "cuda_utils.hpp"
#include "util/debug.hpp"

#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tiledcuda::testing {
using namespace cell;
using namespace cell::copy;
namespace tl = cell::tile_layout;

namespace {
float rand_float(float a = 1e-3, float b = 1) {
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

bool check_correctness(const half* hc1, const float* hc2, int row, int col) {
    int numel = row * col;
    bool pass_unittest = true;
    static const float eps = 5e-2;

#if defined(DEBUG)
    int cut_off = 32;
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
        diff = abs(__half2float(hc1[i]) - hc2[i]);
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
void cublas_hgemm(int m, int n, int k, const __half* A, const __half* B,
                  __half* C, int lda, int ldb, int ldc) {
    __half alf = 1.;
    __half bet = 0.;

    cublasHandle_t handle;
    CublasCheck(cublasCreate(&handle));
    CublasCheck(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alf, A,
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

    /// == 2. configurate tile transfer between global and shared using CuTe ==
    using GlobalA = GlobalTile<Element, tl::RowMajor<kM, kK>>;
    static const bool kSwizzled = true;
    using SharedA = SharedTile<Element, tl::RowMajor<kM, kK>, kSwizzled>;
    using LoadSharedA = GlobalToSharedLoader<SharedA, WarpLayout>;

    using GlobalB = GlobalTile<Element, tl::RowMajor<kN, kK>>;
    using SharedB = SharedTile<Element, tl::RowMajor<kN, kK>>;
    using LoadSharedB = GlobalToSharedLoader<SharedB, WarpLayout>;

    /// === 3. configurate tile transfer between shared and register loader ===
    // shared tile for operand A
    using TileIteratorA = TileIterator<SharedA, TileShape<kM, kChunkK>>;
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

    // register tile for output C
    // calculate register usage for output C
    static constexpr int kCMs = kM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kCNs = kN / kWarpPerCol / BaseShape::kTileSize;

    using RegC =
        RegTile<BaseTileRowMajor<ElementAcc>, tl::RowMajor<kCMs, kCNs>>;
    using GlobalC = GlobalTile<ElementAcc, tl::RowMajor<kM, kN>>;
    using CStorer = copy::RegToGlobalStorer<GlobalC, RegC, WarpLayout>;
};

template <typename Element, typename ElementAcc,                     //
          typename GlobalA, typename SharedA, typename LoadSharedA,  //
          typename GlobalB, typename SharedB, typename LoadSharedB,  //
          typename TileIteratorA, typename RegA, typename LoadRegA,
          typename TileIteratorB, typename RegB, typename LoadRegB,
          typename GlobalC, typename RegC, typename StoreC>
__global__ void test_wmma(const Element* ga, const Element* gb,
                          ElementAcc* gc) {
    GlobalA gA(ga);
    GlobalB gB(gb);

    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* shared_a = reinterpret_cast<Element*>(buf_);
    auto* shared_b = shared_a + TileIteratorA::Tile::kNumel;

    SharedA sA(shared_a);
    SharedB sB(shared_b);

    LoadSharedA loaderA;
    loaderA(gA, sA);

    LoadSharedB loaderB;
    loaderB(gB, sB);
    __copy_async();
    __syncthreads();

    TileIteratorA sAs(shared_a);
    TileIteratorB sBs(shared_b);

    LoadRegA load_rA;
    RegA rA;

    LoadRegB load_rB;
    RegB rB;

    RegC acc;

    for (int k = 0; k < TileIteratorA::sc1; ++k) {
        load_rA(sAs(k), rA);
        load_rB(sBs(k), rB);

        compute::gemm_(rA, rB, acc);
    }
    __syncthreads();

    // store from register to global
    GlobalC gC(gc);
    StoreC store_rC;
    store_rC(acc, gC);
}
}  // namespace

template <const int kM, const int kN, const int kK, typename WarpLayout,
          const int kChunkK>
void run_test() {
    /// unittest for register-level gemm by calling into wmma PTX
    using Element = cutlass::half_t;
    using ElementAcc = float;

    // initialize data
    thrust::host_vector<Element> h_a(kM * kK);
    for (int i = 0; i < h_a.size(); ++i) {
        h_a[i] = static_cast<Element>(rand_float());
    }

    thrust::host_vector<Element> h_b(kK * kN);
    for (int i = 0; i < h_b.size(); ++i) {
        h_b[i] = static_cast<Element>(rand_float());
    }

    thrust::host_vector<ElementAcc> h_c(kM * kN);
    thrust::fill(h_c.begin(), h_c.end(), 0.);

    thrust::device_vector<Element> d_a = h_a;
    thrust::device_vector<Element> d_b = h_b;
    thrust::device_vector<ElementAcc> d_c = h_c;

    /// define the configuration of the test
    using config =
        TestTraits<Element, ElementAcc, kM, kN, kK, WarpLayout, kChunkK>;

    LOG(INFO) << "[" << kM << ", " << kN << ", " << kK << "], warps: ["
              << config::kWarpPerRow << ", " << config::kWarpPerCol
              << "], k_chunk_size: " << kChunkK
              << ", kThreads: " << config::kThreads << std::endl;

    using RegA = typename config::RegA;
    using RegB = typename config::RegB;
    using RegC = typename config::RegC;

    using IteratorA = typename config::TileIteratorA;
    using IteratorB = typename config::TileIteratorB;

#if defined(DEBUG)
    LOG(INFO) << "TileIteratorA: " << IteratorA{} << std::endl
              << "TileIteratorB: " << IteratorB{} << std::endl
              << "RegA: " << RegA{} << std::endl
              << "RegB: " << RegB{} << std::endl
              << "RegC: " << RegC{} << std::endl;
#endif

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(config::kThreads, 1, 1);
    int shm_size = (kM + kN) * kK * sizeof(Element);

    // TODO: Refine this code; there are too many template parameters, making it
    // messy.
    test_wmma<
        Element, ElementAcc, typename config::GlobalA, typename config::SharedA,
        typename config::LoadSharedA, typename config::GlobalB,
        typename config::SharedB, typename config::LoadSharedB, IteratorA, RegA,
        typename config::LoadRegA, IteratorB, RegB, typename config::LoadRegB,
        typename config::GlobalC, RegC, typename config::CStorer>
        <<<dim_grid, dim_block, shm_size>>>(
            thrust::raw_pointer_cast(d_a.data()),
            thrust::raw_pointer_cast(d_b.data()),
            thrust::raw_pointer_cast(d_c.data()));
    cudaDeviceSynchronize();
    h_c = d_c;

    /// unittest for correctness, take cublas as the ground-truth
    thrust::device_vector<__half> d_cublas(kM * kN);
    thrust::fill(d_cublas.begin(), d_cublas.end(), 0.);

    cublas_hgemm(
        kN, kM, kK,
        reinterpret_cast<const __half*>(thrust::raw_pointer_cast(d_b.data())),
        reinterpret_cast<const __half*>(thrust::raw_pointer_cast(d_a.data())),
        reinterpret_cast<__half*>(thrust::raw_pointer_cast(d_cublas.data())),
        kK /*lda*/, kK /*ldb*/, kN /*ldc*/);
    thrust::host_vector<__half> h_cublas = d_cublas;

    EXPECT_TRUE(check_correctness(thrust::raw_pointer_cast(h_cublas.data()),
                                  thrust::raw_pointer_cast(h_c.data()), kM, kN))
        << "Failed test!" << std::endl;
}

TEST(TestGemm, test) {
    // minimal shape for 1 warp
    run_test<16, 16, 16, tl::RowMajor<1, 1>, 16>();
    run_test<32, 32, 64, tl::RowMajor<1, 1>, 16>();
    run_test<32, 32, 32, tl::RowMajor<1, 1>, 32>();

    // minimal shape for 2 warps
    run_test<32, 32, 64, tl::RowMajor<1, 2>, 32>();
    run_test<64, 32, 128, tl::RowMajor<2, 1>, 32>();

    // minimal shape for 2 x 2 warps
    run_test<32, 32, 64, tl::RowMajor<2, 2>, 32>();
    run_test<32, 32, 64, tl::RowMajor<2, 2>, 32>();
    run_test<64, 32, 64, tl::RowMajor<2, 2>, 32>();
    run_test<32, 32, 128, tl::RowMajor<2, 2>, 64>();

    run_test<64, 64, 64, tl::RowMajor<2, 2>, 32>();
    run_test<64, 32, 128, tl::RowMajor<2, 2>, 32>();
}

}  // namespace tiledcuda::testing
