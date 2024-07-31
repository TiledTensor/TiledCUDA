#include "cell/mod.hpp"
#include "cell/test_gemm_utils.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tiledcuda::testing {
using namespace cell;
using namespace cell::copy;
namespace tl = cell::tile_layout;

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

    // const int tid = 10;

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

        // if (thread(tid)) {
        //     printf("\nsA2:\n");
        //     sAs2(k).dump_value();

        //     printf("\nswizzled sA:\n");
        //     __half* data = shared_a2;
        //     for (int i = 0; i < TileIteratorA::Tile::kNumel; ++i) {
        //         printf("%.0f, ", __half2float(data[i]));
        //         if (i && (i + 1) % TileIteratorA::Tile::kCols == 0)
        //             printf("\n");
        //     }
        //     printf("\n");

        //     printf("\nrA:\n");
        //     rA.dump_value();

        //     printf("\nrA2:\n");
        //     rA2.dump_value();
        // }
        load_rB(sBs(k), rB);

        compute::gemm_(rA2, rB, acc);
    }
    __syncthreads();

    StoreRegC store_rC;
    store_rC(acc, sC);
    __syncthreads();

    StoreSharedC store_sC;
    store_sC(sC, gC);
}

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
        // h_af[i] = static_cast<Element>(i);
        h_af[i] = rand_float();
        h_a[i] = static_cast<Element>(h_af[i]);
    }

    thrust::host_vector<Element> h_bf(kK * kN);
    thrust::host_vector<Element> h_b(kK * kN);
    for (int i = 0; i < h_b.size(); ++i) {
        // h_bf[i] = static_cast<Element>(i);
        h_bf[i] = rand_float();
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

    LOG(INFO) << "swizzled shape = [" << Layout2::kSwizzledRow << ", "
              << Layout2::kSwizzledCol << "], "
              << "swizzled strides = [" << Layout2::kSwizzledRowStride << ", "
              << Layout2::kSwizzledColStride << "]" << std::endl;

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

    thrust::host_vector<float> h_cublas = d_cublas;
    EXPECT_TRUE(check_correctness(thrust::raw_pointer_cast(h_cublas.data()),
                                  thrust::raw_pointer_cast(h_c.data()), kM, kN))
        << "Failed test!" << std::endl;
}

TEST(TestGemm, test) {
    // Given that the input matrices are laid out in row-major order, the
    // minimal tile shape is 16x32, considering the entire computational
    // process:
    // 1. In the global to shared memory loading process, threads in a warp are
    //    arranged in an 8x4 fashion, with each thread loading 8 elements along
    //    the column dimension. This requires the minimal tile shape to be 8x32.
    // 2. To optimally utilize the tensor core's capabilities, a minimal tile
    //    shape of 16x16 is required.
    // Combining the above two constraints, the  minimal tile shape is 16x32 for
    // a single warp.
    run_test<16, 32, 32, tl::RowMajor<1, 1>, 32>();

    // more settings
    run_test<32, 32, 64, tl::RowMajor<1, 1>, 64>();
    run_test<32, 32, 32, tl::RowMajor<1, 1>, 32>();

    // minimal shape for 2 warps
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
