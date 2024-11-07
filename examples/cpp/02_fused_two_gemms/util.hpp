#pragma once

#include "util/cuda_timer.hpp"

#include <cublas_v2.h>
#include <float.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <const int kM, const int kN, const int kK, const int kP>
using FusedGemmShape = TileShape<kM, kN, kK, kP>;

float rand_float(float a = 1e-1, float b = 5e-2) {
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

/* In this implementation, A and D are interpreted as being laid out in
   row-major, and B, C is interpreted as being laid out in column-major.

  A and D are laid out in row-major fashion
  B and C are laid out in column-major fashion

  acc[m, n] = A[m, k] @ B[k, n]
    D[m, p] = acc[m, n] @ C[n, p]
*/
float cublas_two_gemms(int kM, int kN, int kK, int kP, int kBatch,
                       const __half* As, const __half* Bs, const __half* Cs,
                       __half* Ds, __half* accs, bool timeit = false,
                       int warmup = 10, int iters = 20) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    __half alf = static_cast<__half>(1.);
    __half bet = static_cast<__half>(0.);

    const __half* A = As;
    const __half* B = Bs;
    const __half* C = Cs;
    __half* acc = accs;
    __half* D = Ds;

    for (int b = 0; b < kBatch; ++b) {
        A += b * kM * kK;
        B += b * kK * kN;
        C += b * kM * kN;
        acc += b * kM * kN;
        D += b * kM * kP;

        // acc   = A @ B
        // acc^T = B^T @ A^T
        // [n, m] = [n, k] @ [k, m]
        cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N /* transb*/, kN, kM, kK,
                    &alf, B, kK, A, kK, &bet, acc, kN);

        // D and acc are laid out in row-major fashion, while C is in column
        // major fashion. Operands of cuBLAS is by default in column fashion.
        // D = acc @ C
        // D^T = C^T @ acc^T; [p, m] = [p, n] @ [n, m]
        cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N /* transb*/, kP, kM, kN,
                    &alf, C, kN, acc, kN, &bet, D, kP);
    }

    float elapsed = 0.;
    if (timeit) {
        for (int i = 0; i < warmup; ++i) {
            A = As;
            B = Bs;
            C = Cs;
            acc = accs;
            D = Ds;
            for (int b = 0; b < kBatch; ++b) {
                A += b * kM * kK;
                B += b * kK * kN;
                C += b * kM * kN;
                acc += b * kM * kN;
                D += b * kM * kP;

                cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N /* transb*/, kN,
                            kM, kK, &alf, B, kK, A, kK, &bet, acc, kN);

                cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N /* transb*/, kP,
                            kM, kN, &alf, C, kN, acc, kN, &bet, D, kP);
            }
        }
        cudaDeviceSynchronize();

        CudaTimer timer;
        timer.start();
        for (int i = 0; i < iters; ++i) {
            A = As;
            B = Bs;
            C = Cs;
            acc = accs;
            D = Ds;
            for (int b = 0; b < kBatch; ++b) {
                A += b * kM * kK;
                B += b * kK * kN;
                C += b * kM * kN;
                acc += b * kM * kN;
                D += b * kM * kP;

                cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N /* transb*/, kN,
                            kM, kK, &alf, B, kK, A, kK, &bet, acc, kN);

                cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N /* transb*/, kP,
                            kM, kN, &alf, C, kN, acc, kN, &bet, D, kP);
            }
        }
        cudaDeviceSynchronize();
        elapsed = timer.stop() / iters;
    }

    cublasDestroy(handle);

    return elapsed;
}

bool check_results(const float* values1, const __half* values2, int numel,
                   float epsilon) {
    bool passed = true;

    float v2 = 0.;

    double total_diff = 0.;
    double max_abs_diff = FLT_MIN;
    double diff = 0.;

    for (int i = 0; i < numel; ++i) {
        v2 = __half2float(values2[i]);
        diff = abs(values1[i] - v2);
        max_abs_diff = max_abs_diff < diff ? diff : max_abs_diff;
        total_diff += diff;

#ifdef DEBUG
        if (diff > epsilon) {
            printf("%d-th value has large differences: %.3f vs. %.3f\n", i,
                   values1[i], v2);
        }
#endif
    }

    double avg_diff = total_diff / numel;
    if (avg_diff > epsilon) passed = false;

    return passed;
}
