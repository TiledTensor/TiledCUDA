#include "common/test_utils.hpp"

namespace tiledcuda::testing {

// FIXME(haruhi): A quick implementation to compare two __half arrays. Refine
// the implementation of necessary unittest utilities.
void CheckResult(const __half* v1, const __half* v2, int64_t numel) {
    float epsilon = 1e-5;

    float a = 0.f;
    float b = 0.f;
    for (int i = 0; i < numel; ++i) {
        a = __half2float(v1[i]);
        b = __half2float(v2[i]);

        EXPECT_NEAR(a, b, epsilon) << "v1[" << i << "] vs. v2[" << i
                                   << "] = " << a << " vs. " << b << std::endl;
    }
}

}  // namespace tiledcuda::testing
