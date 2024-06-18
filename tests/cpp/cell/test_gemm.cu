#include "cell/mod.hpp"
#include "common/test_utils.hpp"
#include "types/tile_iterator.hpp"
#include "types/types.hpp"

#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tiledcuda {

using namespace cell;
using namespace cute;

namespace tl = tile_layout;

namespace {

template <typename DType1, typename DType2>
void check_result1(const DType1* a, const DType1* b, DType2* c, int M, int N,
                   int K, const float* result) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            DType2 s = 0.;
            for (int k = 0; k < K; ++k) s += a[i * K + k] * b[k * N + j];
            c[i * N + j] = s;
        }
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            EXPECT_NEAR(c[i * N + j], result[i * N + j], 1e-2);
        }
    }
}

/// utility function
template <typename Element, const int kRows, const int kCols>
__device__ void print_tile(const Element* data, int delimeter = kCols) {
    if (threadIdx.x || blockIdx.x || blockIdx.y) return;

    for (int i = 0; i < kRows * kCols; ++i) {
        printf("%.2f, ", float(data[i] * 1.0_hf));  // print cutlass::half_t
        if ((i + 1) % delimeter == 0) printf("\n");
    }
    printf("\n");
}

template <typename Element, typename ElementAcc, typename LoadSharedA,
          typename LoadSharedB, typename StoreSharedC, typename TileIteratorA,
          typename TileIteratorB, typename SharedC, typename WarpLayout>
__global__ void test_wmma1(const Element* ga, const Element* gb,
                           ElementAcc* gc) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* shared_a = reinterpret_cast<Element*>(buf_);
    auto* shared_b = shared_a + TileIteratorA::Tile::kNumel;
    auto* shared_c = reinterpret_cast<ElementAcc*>(buf_);

    // transfer data tiles from global to shared
    copy::copy_2d_tile_g2s(ga, shared_a, typename LoadSharedA::SrcLayout{},
                           typename LoadSharedA::DstLayout{},
                           typename LoadSharedA::TiledCopy{});
    copy::copy_2d_tile_g2s(gb, shared_b, typename LoadSharedB::SrcLayout{},
                           typename LoadSharedB::DstLayout{},
                           typename LoadSharedB::TiledCopy{});
    __copy_async();
    __syncthreads();

    Reg<Element, tl::RowMajor<4, 24>> rA;
    Reg<Element, tl::RowMajor<4, 24>> rB;
    Reg<ElementAcc, tl::RowMajor<2, 8>> acc;

    TileIteratorA sAs(shared_a);
    TileIteratorB sBs(shared_b);

    static_assert(TileIteratorA::sc1 == TileIteratorB::sc0,
                  "dimension mismatch!");

    for (int k = 0; k < TileIteratorA::sc1; ++k) {
        copy::copy_tile_s2r(*sAs(_, k), rA, WarpLayout{});
        copy::copy_tile_s2r(*sBs(k, _), rB, WarpLayout{});

        compute::gemm_(rA, rB, acc);
    }
    __syncthreads();

    SharedC sC(shared_c);
    copy::copy_tile_r2s(acc, sC, WarpLayout{});
    __syncthreads();

    copy::copy_2d_tile_s2g(shared_c, gc, typename StoreSharedC::SrcLayout{},
                           typename StoreSharedC::DstLayout{},
                           typename StoreSharedC::TiledCopy{});
}

}  // namespace

namespace testing {

TEST(TestWmma, shape1) {
    // unittest for register-level gemm by calling into wmma PTX
    using Element = cutlass::half_t;
    using ElementAcc = float;

    using WarpLayout = tl::RowMajor<2, 2>;
    static const int kThreads = tl::get_numel<WarpLayout> * 32;

    LOG(INFO) << "kThreads: " << kThreads << std::endl;

    // M, N, K for shared memory tile
    const int M = 64;
    const int N = 64;
    const int K = 128;

    thrust::host_vector<Element> h_a(M * K);
    thrust::host_vector<Element> h_b(K * N);
    thrust::host_vector<ElementAcc> h_c(M * N);

    for (int i = 0; i < h_a.size(); ++i)
        h_a[i] = static_cast<Element>(i % 2048 / 100.);

    for (int i = 0; i < h_b.size(); ++i)
        h_b[i] = static_cast<Element>(i % 2048 / 100.);

    thrust::fill(h_c.begin(), h_c.end(), 0.);

    thrust::device_vector<Element> d_a = h_a;
    thrust::device_vector<Element> d_b = h_b;
    thrust::device_vector<ElementAcc> d_c = h_c;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);

    int size_ab = (M + N) * K * sizeof(Element);
    int size_c = M * N * sizeof(ElementAcc);
    int shm_size = size_ab > size_c ? size_ab : size_c;

    using SharedA = Shared<Element, tl::RowMajor<M, K>>;
    using TileIteratorA = TileIterator<SharedA, TileShape<32, 32>>;
    LOG(INFO) << "TileIteratorA: [" << TileIteratorA::Tile::kRows << ", "
              << TileIteratorA::Tile::kCols
              << "]; numel = " << TileIteratorA::Tile::kNumel << std::endl
              << "sc0 = " << TileIteratorA::sc0
              << ", sc1 = " << TileIteratorA::sc1 << std::endl;

    using SharedB = Shared<Element, tl::RowMajor<K, N>>;
    using TileIteratorB = TileIterator<SharedB, TileShape<32, 32>>;
    LOG(INFO) << "TileIteratorB: sc0 = " << TileIteratorB::sc0
              << ", sc1 = " << TileIteratorB::sc1 << std::endl;

    using SharedC = Shared<ElementAcc, tl::RowMajor<M, N>>;

    // for global to shared memory copy using CuTe
    using LoadSharedA = traits::G2S2DCopyTraits<Element, M, K, M, K, kThreads,
                                                false /*use swizzle*/>;
    using LoadSharedB = traits::G2S2DCopyTraits<Element, N, K, N, K, kThreads,
                                                false /*use swizzle*/>;
    // transfer operand C from shared memory to global memory
    using StoreSharedC =
        traits::S2G2DCopyTraits<ElementAcc, M, N, M, N, kThreads,
                                false /*use swizzle*/>;

    test_wmma1<Element, ElementAcc, LoadSharedA, LoadSharedB, StoreSharedC,
               TileIteratorA, TileIteratorB, SharedC, WarpLayout>
        <<<dim_grid, dim_block, shm_size>>>(
            thrust::raw_pointer_cast(d_a.data()),
            thrust::raw_pointer_cast(d_b.data()),
            thrust::raw_pointer_cast(d_c.data()));

    cudaDeviceSynchronize();

    thrust::host_vector<float> h_c1;
    h_c1 = d_c;

    // {  // ground truth

    //     check_result1(thrust::raw_pointer_cast(h_a.data()),
    //                   thrust::raw_pointer_cast(h_b.data()),
    //                   thrust::raw_pointer_cast(h_c.data()), M, N, K,
    //                   thrust::raw_pointer_cast(h_c1.data()));
    // }
}

}  // namespace testing
}  // namespace tiledcuda
