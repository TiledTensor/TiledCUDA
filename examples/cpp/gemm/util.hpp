#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cfloat>

template <const int kM, const int kN, const int kK>
using GemmShape = TileShape<kM, kN, kK>;

float rand_float(float a = 1e-3, float b = 1) {
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

// In this implementation, A and C are interpreted as being laid out in
// row-major, and B is interpreted as being laid out in column-major.
void naive_gemm(int kM, int kN, int kK,  //
                const __half* A, const __half* B, float* C) {
    for (int i = 0; i < kM; ++i) {
        for (int j = 0; j < kN; ++j) {
            float s = 0.;
            for (int k = 0; k < kK; ++k) {
                s += __half2float(A[i * kK + k]) * __half2float(B[k + kK * j]);
            }
            C[i * kN + j] = s;
        }
    }
}

bool check_results(const float* values1, const float* values2, int numel) {
    bool passed = true;
    const float epsilon = 5e-2;

    double total_diff = 0.;
    double max_abs_diff = FLT_MIN;
    double diff = 0.;

    for (int i = 0; i < numel; ++i) {
        diff = fabs(values1[i] - values2[i]);
        max_abs_diff = max_abs_diff < diff ? diff : max_abs_diff;
        total_diff += diff;

#ifdef DEBUG
        if (diff > epsilon) {
            printf("the %d-th value differs: %.2f vs. %.2f\n", i, values1[i],
                   values2[i]);
        }
#endif
    }

    double avg_diff = total_diff / numel;
    if (avg_diff > epsilon) passed = false;

    return passed;
}
