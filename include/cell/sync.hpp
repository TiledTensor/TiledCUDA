#pragma once
#include "cuda_utils.hpp"

namespace tiledcuda::cell {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#define CP_ASYNC_ENABLED
#endif

template <int N>
__device__ void wait_group() {
#if defined(CP_ASYNC_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

__device__ void commit_copy_group() {
#if defined(CP_ASYNC_ENABLED)
    cute::cp_async_fence();
#endif
}

__device__ void __copy_async() {
    commit_copy_group();
    wait_group<0>();
}

}  // namespace tiledcuda::cell