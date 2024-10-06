#pragma once

#include "util/cuda_timer.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cfloat>

float rand_float(float a = 1e-4, float b = 1e-2) {
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

float cublas_hgemm(int64_t kM, int64_t kN, int64_t kK,  // problem shape
                   const __half* A, const __half* B, __half* C,
                   bool timeit = false) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    __half alf = static_cast<__half>(1.);
    __half bet = static_cast<__half>(0.);

    float elapsed = 0.;

    if (timeit) {
        for (int i = 0; i < 5; ++i) {
            cublasHgemm(handle, CUBLAS_OP_T /* transb*/, CUBLAS_OP_N, kN, kM,
                        kK, &alf, B, kK, A, kK, &bet, C, kN);
        }
        cudaDeviceSynchronize();

        int iters = 20;
        CudaTimer timer;
        timer.start();
        for (int i = 0; i < iters; ++i) {
            cublasHgemm(handle, CUBLAS_OP_T /* transb*/, CUBLAS_OP_N, kN, kM,
                        kK, &alf, B, kK, A, kK, &bet, C, kN);
        }
        cudaDeviceSynchronize();
        elapsed = timer.stop() / iters;
    } else {
        cublasHgemm(handle, CUBLAS_OP_T /* transb*/, CUBLAS_OP_N, kN, kM, kK,
                    &alf, B, kK, A, kK, &bet, C, kN);
    }
    cudaDeviceSynchronize();

    cublasDestroy(handle);
    return elapsed;
}

#define DEBUG

bool check_results(const __half* values1, const __half* values2, int numel) {
    bool passed = true;
    const float epsilon = 1e-1;

    double total_diff = 0.;
    double max_abs_diff = FLT_MIN;
    double diff = 0.;

#ifdef DEBUG
    int cut_off = 128;
    printf("ground truth:\n");
    for (int i = 0; i < cut_off; ++i) {
        printf("%.3f, ", __half2float(values2[i]));
        if (i && (i + 1) % 16 == 0) printf("\n");
    }
    printf("\ncomputed values:\n");
    for (int i = 0; i < cut_off; ++i) {
        printf("%.3f, ", __half2float(values1[i]));
        if (i && (i + 1) % 16 == 0) printf("\n");
    }
#endif

    for (int i = 0; i < numel; ++i) {
        float v1 = __half2float(values1[i]);
        float v2 = __half2float(values2[i]);
        diff = fabs(v1 - v2);
        max_abs_diff = max_abs_diff < diff ? diff : max_abs_diff;
        total_diff += diff;

#ifdef DEBUG
        if (diff > epsilon) {
            printf("the %d-th value differs (%.4f): %.4f vs. %.4f\n", i, diff,
                   v1, v2);
        }
#endif
    }

    double avg_diff = total_diff / numel;
    if (avg_diff > epsilon) passed = false;

    return passed;
}
