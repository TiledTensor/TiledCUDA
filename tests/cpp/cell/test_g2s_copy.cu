#include "cell/copy/mod.hpp"
#include "cell/sync.hpp"
#include "cell/traits/copy.hpp"
#include "common/test_utils.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tiledcuda::testing {

namespace {
// the host function to call the device copy function
template <typename Element, typename G2STraits, typename S2GTraits>
__global__ void copy_g2s(const Element* src, Element* trg) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);

    // transfer a 2D data tile from global memory to shared memory.
    cell::copy::copy_2d_tile_g2s(src, buf, typename G2STraits::SrcLayout{},
                                 typename G2STraits::DstLayout{},
                                 typename G2STraits::TiledCopy{});
    cell::__copy_async();
    __syncthreads();

    // transfer a 2D data tile from shared memory to global memory.
    cell::copy::copy_2d_tile_s2g(buf, trg, typename S2GTraits::SrcLayout{},
                                 typename S2GTraits::DstLayout{},
                                 typename S2GTraits::TiledCopy{});
    cell::__copy_async();
    __syncthreads();
}
}  // namespace

TEST(GlobalToSharedCopy, test1) {
    namespace traits = tiledcuda::cell::traits;

    // The simple test case for 2D copy. Copy a 16x32 matrix from global
    // memory to shared memory using a single warp.
    // NOTE: This unitttest represents the minimum shape and threads in a
    // thread block allowed for a 2D copy operation.
    using Element = cutlass::half_t;

    static constexpr int kRows = 16;
    static constexpr int kCols = 32;

    static constexpr int kShmRows = kRows;
    static constexpr int kShmCols = kCols;

    // threads are arranged as 8 x 4 to perform 2D copy
    static const int kThreads = 32;

    int numel = kRows * kCols;
    thrust::host_vector<Element> h_A(numel);
    srand(42);
    for (int i = 0; i < h_A.size(); ++i) {
        h_A[i] = __float2half(10 * (rand() / float(RAND_MAX)) - 5);
    }

    // copy data from host to device
    thrust::device_vector<Element> d_A = h_A;
    thrust::device_vector<Element> d_B(numel);
    thrust::fill(d_B.begin(), d_B.end(), static_cast<Element>(0.));

    int m = CeilDiv<kRows, kShmRows>;
    int n = CeilDiv<kCols, kShmCols>;
    dim3 dim_grid(m, n);
    dim3 dim_block(kThreads);

    using G2SCopyTraits = traits::G2S2DCopyTraits<Element, kRows, kCols,
                                                  kShmRows, kShmCols, kThreads>;
    using S2GCopyTraits = traits::S2G2DCopyTraits<Element, kRows, kCols,
                                                  kShmRows, kShmCols, kThreads>;

    int shm_size = kShmRows * kShmCols * sizeof(Element);
    copy_g2s<Element, G2SCopyTraits, S2GCopyTraits>
        <<<dim_grid, dim_block, shm_size>>>(
            thrust::raw_pointer_cast(d_A.data()),
            thrust::raw_pointer_cast(d_B.data()));
    cudaDeviceSynchronize();

    // check correctness
    thrust::host_vector<Element> h_B(numel);
    h_B = d_B;

    assert_equal(
        reinterpret_cast<__half*>(thrust::raw_pointer_cast(h_A.data())),
        reinterpret_cast<__half*>(thrust::raw_pointer_cast(h_B.data())), numel);
}

}  // namespace tiledcuda::testing
