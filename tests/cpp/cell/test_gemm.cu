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

#define DEBUG
bool check_correctness(const float* hc1, const float* hc2, int row, int col) {
    int numel = row * col;
    bool pass_unittest = true;
    static const float eps = 5e-2;
    // static const float eps = 1.;

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

    // test swizzled layout
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

template <typename Element, typename ElementAcc, typename GlobalA,
          typename SharedA2, typename LoadSharedA2,  // test swizzled layout
          typename SharedA, typename LoadSharedA,    //
          typename GlobalB, typename SharedB, typename LoadSharedB,
          typename GlobalC, typename SharedC, typename StoreSharedC,
          typename TileIteratorA2,  //
          typename TileIteratorA, typename RegA, typename LoadRegA,
          typename TileIteratorB, typename RegB, typename LoadRegB,
          typename RegC, typename StoreRegC>
__global__ void test_gemm(const Element* ga, const Element* gb,
                          ElementAcc* gc) {
    GlobalA gA(ga);
    GlobalB gB(gb);
    GlobalC gC(gc);

    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* shared_a = reinterpret_cast<Element*>(buf_);
    auto* shared_a2 =
        reinterpret_cast<Element*>(shared_a + TileIteratorA::Tile::kNumel);
    auto* shared_b = shared_a2 + TileIteratorA2::Tile::kNumel;
    auto* shared_c = reinterpret_cast<ElementAcc*>(buf_);

    SharedA sA(shared_a);
    SharedB sB(shared_b);
    SharedC sC(shared_c);

    LoadSharedA loaderA;
    LoadSharedB loaderB;

    // debug swizzled shared memory
    SharedA2 sA2(shared_a2);
    LoadSharedA2 loaderA2;

    loaderA(gA, sA);
    loaderA2(gA, sA2);
    loaderB(gB, sB);
    __copy_async();
    __syncthreads();

    if (thread(27)) {
        // printf("gA:\n");
        // gA.dump_value();

        printf("\nsA:\n");
        sA.dump_value();

        // printf("\nsA2:\n");
        // sA2.dump_value();

        printf("\nswizzled sA:\n");
        __half* data = shared_a2;
        for (int i = 0; i < TileIteratorA::Tile::kNumel; ++i) {
            printf("%.0f, ", __half2float(data[i]));
            if (i && (i + 1) % TileIteratorA::Tile::kCols == 0) printf("\n");
        }
        printf("\n");
    }

    TileIteratorA sAs(shared_a);
    TileIteratorA2 sAs2(shared_a2);
    TileIteratorB sBs(shared_b);

    LoadRegA load_rA;
    RegA rA;
    RegA rA2;

    LoadRegB load_rB;
    RegB rB;

    RegC acc;

    for (int k = 0; k < TileIteratorA::sc1; ++k) {
        load_rA(sAs(k), rA);
        load_rA(sAs2(k), rA2);

        if (thread(27)) {
            printf("\nk: %d\nrA:\n", k);
            rA.dump_value();

            printf("\nrA2:\n");
            rA2.dump_value();
        }

        load_rB(sBs(k), rB);

        compute::gemm_(rA, rB, acc);
    }
    __syncthreads();

    StoreRegC store_rC;
    store_rC(acc, sC);
    __syncthreads();

    StoreSharedC store_sC;
    store_sC(sC, gC);
}
}  // namespace

template <const int kM, const int kN, const int kK, typename WarpLayout,
          const int kChunkK>
void run_test() {
    /// unittest for register-level gemm by calling into wmma PTX
    using Element = __half;
    using ElementAcc = float;

    // initialize data
    thrust::host_vector<ElementAcc> h_af(kM * kK);
    thrust::host_vector<ElementAcc> h_a(kM * kK);
    for (int i = 0; i < h_a.size(); ++i) {
        h_af[i] = static_cast<Element>(i);
        // h_af[i] = rand_float();
        h_a[i] = static_cast<Element>(h_af[i]);
    }

    thrust::host_vector<Element> h_bf(kK * kN);
    thrust::host_vector<Element> h_b(kK * kN);
    for (int i = 0; i < h_b.size(); ++i) {
        h_bf[i] = static_cast<Element>(i);
        // h_bf[i] = rand_float();
        h_b[i] = static_cast<Element>(h_bf[i]);
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
    using IteratorA2 = typename config::TileIteratorA2;
    using IteratorB = typename config::TileIteratorB;

    using Layout1 = typename config::Layout1;
    using Layout2 = typename config::Layout2;

    Layout1 layout;
    Layout2 swizzled;

    // std::cout << "RowMajor: " << std::endl;
    // for (int i = 0; i < Layout2::kRows; ++i) {
    //     for (int j = 0; j < Layout2::kCols; ++j) {
    //         std::cout << layout(i, j) << ", ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "Swizzled: " << std::endl;
    // for (int i = 0; i < Layout2::kRows; ++i) {
    //     for (int j = 0; j < Layout2::kCols; ++j) {
    //         std::cout << swizzled(i, j) << ", ";
    //     }
    //     std::cout << std::endl;
    // }

#if defined(DEBUG)
    LOG(INFO) << "TileIteratorA: " << IteratorA{} << std::endl
              << "TileIteratorB: " << IteratorB{} << std::endl
              << "RegA: " << RegA{} << std::endl
              << "RegB: " << RegB{} << std::endl
              << "RegC: " << RegC{} << std::endl;
#endif

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(config::kThreads, 1, 1);
    int size_ab = (kM + kN) * kK * sizeof(Element);
    int size_a = kM * kK * sizeof(Element);  // store swizzled A
    int size_c = kM * kN * sizeof(ElementAcc);

    int size_inputs = size_ab + size_a;
    int shm_size = size_inputs > size_c ? size_inputs : size_c;

    // TODO: Refine this code; there are too many template parameters, making it
    // messy.
    test_gemm<Element, ElementAcc, typename config::GlobalA,
              typename config::SharedA2, typename config::LoadSharedA2,
              typename config::SharedA, typename config::LoadSharedA,
              typename config::GlobalB,                                 //
              typename config::SharedB, typename config::LoadSharedB,   //
              typename config::GlobalC,                                 //
              typename config::SharedC, typename config::StoreSharedC,  //
              IteratorA2,                                               //
              IteratorA, RegA, typename config::LoadRegA,               //
              IteratorB, RegB, typename config::LoadRegB, RegC,         //
              typename config::StoreRegC><<<dim_grid, dim_block, shm_size>>>(
        thrust::raw_pointer_cast(d_a.data()),
        thrust::raw_pointer_cast(d_b.data()),
        thrust::raw_pointer_cast(d_c.data()));
    cudaDeviceSynchronize();
    h_c = d_c;

    /// unittest for correctness, take cublas as the ground-truth
    thrust::device_vector<float> d_af = h_af;
    thrust::device_vector<float> d_bf = h_bf;
    thrust::device_vector<float> d_cublas(kM * kN);
    thrust::fill(d_cublas.begin(), d_cublas.end(), 0.);

    cublas_sgemm(kN, kM, kK, thrust::raw_pointer_cast(d_bf.data()),
                 thrust::raw_pointer_cast(d_af.data()),
                 thrust::raw_pointer_cast(d_cublas.data()), kK /*lda*/,
                 kK /*ldb*/, kN /*ldc*/);

    // thrust::host_vector<float> h_cublas = d_cublas;
    // EXPECT_TRUE(check_correctness(thrust::raw_pointer_cast(h_cublas.data()),
    //                               thrust::raw_pointer_cast(h_c.data()), kM,
    //                               kN))
    //     << "Failed test!" << std::endl;
}

TEST(TestGemm, test) {
    // minimal shape for 1 warp
    run_test<16, 32, 32, tl::RowMajor<1, 1>, 16>();
    // run_test<32, 32, 64, tl::RowMajor<1, 1>, 16>();
    // run_test<32, 32, 32, tl::RowMajor<1, 1>, 32>();
    // // minimal shape for 2 warps
    // run_test<32, 32, 64, tl::RowMajor<1, 2>, 32>();
    // run_test<64, 32, 128, tl::RowMajor<2, 1>, 32>();

    // // minimal shape for 2 x 2 warps
    // run_test<32, 32, 64, tl::RowMajor<2, 2>, 32>();
    // run_test<32, 32, 64, tl::RowMajor<2, 2>, 32>();
    // run_test<64, 32, 64, tl::RowMajor<2, 2>, 32>();
    // run_test<32, 32, 128, tl::RowMajor<2, 2>, 64>();

    // run_test<64, 64, 64, tl::RowMajor<2, 2>, 32>();
    // run_test<64, 32, 128, tl::RowMajor<2, 2>, 32>();
}

}  // namespace tiledcuda::testing
