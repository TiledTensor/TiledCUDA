#pragma once

#include "cuda_utils.hpp"

namespace tiledcuda::cell {

template <int N>
DEVICE void wait_group() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

DEVICE void commit_copy_group() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    cute::cp_async_fence();
#endif
}

DEVICE void __copy_async() {
    commit_copy_group();
    wait_group<0>();
}

}  // namespace tiledcuda::cell
