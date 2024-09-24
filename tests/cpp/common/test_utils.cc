#include "common/test_utils.hpp"

namespace tiledcuda::testing {

// FIXME(haruhi): A quick implementation to compare two __half arrays. Refine
// the implementation of necessary unittest utilities.
template <>
void assert_equal(const __half* v1, const __half* v2, int64_t numel,
                  float epsilon) {
    float a = 0.f;
    float b = 0.f;
    for (int i = 0; i < numel; ++i) {
        a = __half2float(v1[i]);
        b = __half2float(v2[i]);

        EXPECT_NEAR(a, b, epsilon) << "v1[" << i << "] vs. v2[" << i
                                   << "] = " << a << " vs. " << b << std::endl;
    }
}

template <>
void assert_equal(const float* v1, const float* v2, int64_t numel,
                  float epsilon) {
    for (int i = 0; i < numel; ++i)
        EXPECT_NEAR(v1[i], v2[i], epsilon)
            << "v1[" << i << "] vs. v2[" << i << "] = " << v1[i] << " vs. "
            << v2[i] << std::endl;
}

}  // namespace tiledcuda::testing
