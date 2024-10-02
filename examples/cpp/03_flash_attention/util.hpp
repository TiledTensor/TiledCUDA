#pragma once

#include "flash_attn_cpu.hpp"
#include "util/debug.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

float rand_float(float a = 1e-1, float b = 5e-2) {
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

bool check_results(const __half* values1, const __half* values2, int numel) {
    bool passed = true;
    const float epsilon = 1e-1;

    for (int i = 0; i < numel; ++i) {
        if (fabs(__half2float(values1[i]) - __half2float(values2[i])) >
            epsilon) {
            printf("%d-th value differs: %.3f vs. %.3f\n", i,
                   __half2float(values1[i]), __half2float(values2[i]));
            passed = false;
            break;
        }
    }
    return passed;
}
