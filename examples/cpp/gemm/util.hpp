#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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
    const float epsilon = 1e-3;

    for (int i = 0; i < numel; ++i) {
        if (fabs(values1[i] - values2[i]) > epsilon) {
            printf("Diff: %.2f vs. %.2f\n", values1[i], values2[i]);
            passed = false;
            break;
        }
    }
    return passed;
}
