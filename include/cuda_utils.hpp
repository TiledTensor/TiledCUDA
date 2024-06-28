#pragma once

#include "config.hpp"

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>

namespace tiledcuda {

template <int a, int b>
inline constexpr int CeilDiv = (a + b - 1) / b;  // for compile-time values

const char* cublasGetErrorString(cublasStatus_t status);

inline void __cudaCheck(const cudaError err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s(%d): CUDA error: %s.\n", file, line,
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
#define CudaCheck(call) __cudaCheck(call, __FILE__, __LINE__)

inline void __cublasCheck(const cublasStatus_t err, const char* file,
                          int line) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s(%d): Cublas error: %s.\n", file, line,
                cublasGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
#define CublasCheck(call) __cublasCheck(call, __FILE__, __LINE__)

}  // namespace tiledcuda
