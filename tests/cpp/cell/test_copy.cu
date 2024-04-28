#include "tiled_cuda_test.h"

#include <cutlass/half.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tiledcuda {
namespace testing {

TEST(TestG2SCopy, Copy2DTile) {
    using Element = cutlass::half_t;
    static constexpr int kRows = 16 * 2;
    static constexpr int kCols = 16 * 3;

    static constexpr int kShmRows = 16;
    static constexpr int kShmCols = 16;

    int numel = kRows * kCols;
    thrust::host_vector<Element> h_A(numel);
    srand(42);
    for (int i = 0; i < h_A.size(); ++i) {
        h_A[i] = __float2half(10 * (rand() / float(RAND_MAX)) - 5);
    }
    for (int i = 0; i < kRows; ++i) {
        for (int j = 0; j < kCols; ++j) {
            LOG(INFO) << h_A[i * kCols + j];
        }
    }

    // copy data from host to device
    thrust::device_vector<Element> d_A = h_A;
    thrust::device_vector<Element> d_B(numel);
    thrust::fill(d_B.begin(), d_B.end(), static_cast<Element>(0.));
}

}  // namespace testing
}  // namespace tiledcuda
