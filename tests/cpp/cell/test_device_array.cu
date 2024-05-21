#include "cell/copy/mod.hpp"
#include "common/test_utils.hpp"
#include "types/types.hpp"

#include <assert.h>

namespace tiledcuda {

using namespace cell;
using namespace cute;

namespace testing {

namespace {
__global__ void test_dev_array() {
    using IntArray = DevArray<int, 16>;
    IntArray arr;

    assert(IntArray::kSize == 16);

    for (int i = 0; i < IntArray::kSize; ++i) arr[i] = i;

    for (int i = 0; i < IntArray::kSize; ++i) assert(arr[i] == i);
}
}  // namespace

TEST(TestDeviceArray, test_dev_array) {
    test_dev_array<<<1, 1>>>();
    cudaDeviceSynchronize();
}

}  // namespace testing
}  // namespace tiledcuda
