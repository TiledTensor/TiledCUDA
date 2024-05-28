#pragma once

#if defined(__CUDA_ARCH__)
#define HOST_DEVICE __forceinline__ __host__ __device__
#define DEVICE __forceinline__ __device__
#define HOST __forceinline__ __host__
#else
#define HOST_DEVICE inline
#define DEVICE inline
#define HOST inline
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#define CP_ASYNC_SM80_ENABLED
#endif
