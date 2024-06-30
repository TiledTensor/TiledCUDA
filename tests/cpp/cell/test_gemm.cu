#include "cell/mod.hpp"
#include "common/test_utils.hpp"
#include "cuda_utils.hpp"
#include "util/debug.hpp"

#include <cublas_v2.h>
#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tiledcuda {

using namespace cell;
using namespace cell::copy;
namespace tl = tile_layout;

using namespace cute;

namespace {

float rand_float(float a = 1e-3, float b = 1) {
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

#define DEBUG_PRINT

void check_correctness(const half* hc1, const float* hc2, int row, int col) {
    int numel = row * col;

#ifdef DEBUG_PRINT
    printf("cublas:\n");
    for (int i = 0; i < 128; i++) {
        printf("%.2f, ", __half2float(hc1[i]));
        if (i & (i + 1) % 8 == 0) printf("\n");
    }

    printf("\nours:\n");
    for (int i = 0; i < 128; i++) {
        printf("%.2f, ", hc2[i]);
        if (i & (i + 1) % 8 == 0) printf("\n");
    }
    printf("\n");
#endif

    bool pass_unittest = true;
    for (int i = 0; i < numel; ++i) {
        float diff = __half2float(hc1[i]) - hc2[i];
        if (diff > 1e-2) {
            printf("Error results [%d], Expected: %.3f, Got: %.3f\n", i,
                   __half2float(hc1[i]), hc2[i]);
            pass_unittest = false;
        }
    }
    assert(pass_unittest);
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

template <typename Element, typename ElementAcc,  //
          const int M, const int N, const int K>
struct TestTraits {
    using WarpLayout = tl::RowMajor<2, 2>;
    static constexpr int warp_per_row = tl::num_rows<WarpLayout>;
    static constexpr int warp_per_col = tl::num_cols<WarpLayout>;

    static const int kThreads = tl::get_numel<WarpLayout> * 32;

    // for global to shared memory copy using CuTe
    using LoadSharedA = traits::G2S2DCopyTraits<Element, M, K, M, K, kThreads,
                                                false /*use swizzle*/>;
    using LoadSharedB = traits::G2S2DCopyTraits<Element, N, K, N, K, kThreads,
                                                false /*use swizzle*/>;
    // transfer operand C from shared memory to global memory
    using StoreSharedC =
        traits::S2G2DCopyTraits<ElementAcc, M, N, M, N, kThreads,
                                false /*use swizzle*/>;

    // ============= shared to register loader =================
    static constexpr int chunk_row = 32;
    static constexpr int chunk_col = 32;
    using ChunkShape = TileShape<chunk_row, chunk_col>;

    static constexpr int reg_tile_row = 2;
    static constexpr int reg_tile_col = 4;

    using SharedA = SharedTile<Element, tl::RowMajor<M, K>>;
    using TileIteratorA = SharedTileIterator<SharedA, ChunkShape>;

    static constexpr int rA_row = M / warp_per_row / 16 * reg_tile_row;
    static constexpr int rA_col = K / 16 * reg_tile_col;

    using RegA = RegTile<Element, tl::RowMajor<rA_row, rA_col>>;
    using LoadRegA =
        SharedToRegLoader<RegA, WarpLayout, WarpReuse::RowReuseCont,
                          CopyInst::LoadMat>;

    using SharedB = SharedTile<Element, tl::ColMajor<K, N>>;

    static constexpr int rB_row = K / 16 * reg_tile_col;
    static constexpr int rB_col = N / warp_per_col / 16 * reg_tile_row;

    using RegB = RegTile<Element, tl::RowMajor<rB_col, rB_row>>;

    using TileIteratorB = SharedTileIterator<SharedB, ChunkShape>;
    using LoadRegB =
        SharedToRegLoader<RegB, WarpLayout, WarpReuse::ColReuseCont,
                          CopyInst::LoadMat>;

    static_assert(TileIteratorA::sc1 == TileIteratorB::sc0,
                  "mismatched K dimension!");

    // ============= register to shared storer =================
    using SharedC = SharedTile<ElementAcc, tl::RowMajor<M, N>>;

    static constexpr int rC_row = M / warp_per_row / 16 * reg_tile_row;
    static constexpr int rC_col = N / warp_per_col / 16 * reg_tile_col;
    using RegC = RegTile<ElementAcc, tl::RowMajor<rC_row, rC_col>>;

    using StoreRegC =
        RegToSharedStorer<RegC, WarpLayout, RegLayout::WMMA_m16n16k16,
                          CopyInst::LoadS32>;
};

template <typename Element, typename ElementAcc,                              //
          typename LoadSharedA, typename LoadSharedB, typename StoreSharedC,  //
          typename TileIteratorA, typename RegA, typename LoadRegA,
          typename TileIteratorB, typename RegB, typename LoadRegB,
          typename SharedC, typename RegC, typename StoreRegC>
__global__ void test_wmma(const Element* ga, const Element* gb, ElementAcc* gc,
                          LoadRegA& load_rA, LoadRegB& load_rB,
                          StoreRegC& store_rC) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* shared_a = reinterpret_cast<Element*>(buf_);
    auto* shared_b = shared_a + TileIteratorA::Tile::kNumel;
    auto* shared_c = reinterpret_cast<ElementAcc*>(buf_);
    SharedC sC(shared_c);

    copy_2d_tile_g2s(ga, shared_a, typename LoadSharedA::SrcLayout{},
                     typename LoadSharedA::DstLayout{},
                     typename LoadSharedA::TiledCopy{});
    copy_2d_tile_g2s(gb, shared_b, typename LoadSharedB::SrcLayout{},
                     typename LoadSharedB::DstLayout{},
                     typename LoadSharedB::TiledCopy{});
    __copy_async();
    __syncthreads();

    TileIteratorA sAs(shared_a);
    TileIteratorB sBs(shared_b);

    RegA rA;
    RegB rB;
    RegC acc;

    for (int k = 0; k < TileIteratorA::sc1; ++k) {
        auto sA = sAs(k);
        auto sB = sBs(k);

        load_rA(sA, rA);
        load_rB(sB, rB);

        compute::gemm_(rA, rB, acc);
    }
    __syncthreads();

    store_rC(acc, sC);
    __syncthreads();

    copy_2d_tile_s2g(shared_c, gc, typename StoreSharedC::SrcLayout{},
                     typename StoreSharedC::DstLayout{},
                     typename StoreSharedC::TiledCopy{});
}

}  // namespace

namespace testing {

template <const int M, const int N, const int K>
void run_test() {
    /// unittest for register-level gemm by calling into wmma PTX
    using Element = cutlass::half_t;
    using ElementAcc = float;

    // initialize data
    thrust::host_vector<Element> h_a(M * K);
    for (int i = 0; i < h_a.size(); ++i)
        h_a[i] = static_cast<Element>(rand_float());

    thrust::host_vector<Element> h_b(K * N);
    for (int i = 0; i < h_b.size(); ++i)
        h_b[i] = static_cast<Element>(rand_float());

    thrust::host_vector<ElementAcc> h_c(M * N);
    thrust::fill(h_c.begin(), h_c.end(), 0.);

    thrust::device_vector<Element> d_a = h_a;
    thrust::device_vector<Element> d_b = h_b;
    thrust::device_vector<ElementAcc> d_c = h_c;

    /// define the configuration of the test
    using config = TestTraits<Element, ElementAcc, M, N, K>;

    LOG(INFO) << "kThreads: " << config::kThreads << std::endl;
    LOG(INFO) << "TileIteratorA: [" << config::TileIteratorA::Tile::kRows
              << ", " << config::TileIteratorA::Tile::kCols
              << "]; numel = " << config::TileIteratorA::Tile::kNumel
              << ", sc0 = " << config::TileIteratorA::sc0
              << ", sc1 = " << config::TileIteratorA::sc1 << std::endl;
    LOG(INFO) << "TileIteratorB: [" << config::TileIteratorB::Tile::kRows
              << ", " << config::TileIteratorB::Tile::kCols
              << "]; numel = " << config::TileIteratorB::Tile::kNumel
              << ", sc0 = " << config::TileIteratorB::sc0
              << ", sc1 = " << config::TileIteratorB::sc1 << std::endl
              << std::endl;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(config::kThreads, 1, 1);
    int size_ab = (M + N) * K * sizeof(Element);
    int size_c = M * N * sizeof(ElementAcc);
    int shm_size = size_ab > size_c ? size_ab : size_c;

    typename config::LoadRegA load_rA;
    typename config::LoadRegB load_rB;
    typename config::StoreRegC store_rC;

    // TODO: Refine this code; there are too many template parameters, making it
    // messy.
    test_wmma<Element, ElementAcc, typename config::LoadSharedA,
              typename config::LoadSharedB, typename config::StoreSharedC,  //
              typename config::TileIteratorA, typename config::RegA,
              typename config::LoadRegA,  //
              typename config::TileIteratorB, typename config::RegB,
              typename config::LoadRegB,  //
              typename config::SharedC, typename config::RegC,
              typename config::StoreRegC  //
              ><<<dim_grid, dim_block, shm_size>>>(
        thrust::raw_pointer_cast(d_a.data()),
        thrust::raw_pointer_cast(d_b.data()),
        thrust::raw_pointer_cast(d_c.data()), load_rA, load_rB, store_rC);
    cudaDeviceSynchronize();
    h_c = d_c;

    /// unittest for correctness, take cublas as the ground-truth
    thrust::device_vector<__half> d_cublas(M * N);
    thrust::fill(d_cublas.begin(), d_cublas.end(), 0.);

    cublas_hgemm(
        N, M, K,
        reinterpret_cast<const __half*>(thrust::raw_pointer_cast(d_b.data())),
        reinterpret_cast<const __half*>(thrust::raw_pointer_cast(d_a.data())),
        reinterpret_cast<__half*>(thrust::raw_pointer_cast(d_cublas.data())),
        K /*lda*/, K /*ldb*/, N /*ldc*/);

    thrust::host_vector<__half> h_cublas = d_cublas;

    check_correctness(thrust::raw_pointer_cast(h_cublas.data()),
                      thrust::raw_pointer_cast(h_c.data()), M, N);
}

TEST(TestWmma, TestGemm) { run_test<32, 32, 32>(); }

}  // namespace testing
}  // namespace tiledcuda
