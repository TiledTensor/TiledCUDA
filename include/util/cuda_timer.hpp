#pragma once

#include "cuda_utils.hpp"

namespace tiledcuda {

/// @brief: Cuda timer to measure the time taken by a kernel.
/// Usage:
///    CudaTimer timer;
///    timer.start();
///        ...
///    float time = timer.stop();
class CudaTimer {
  public:
    CudaTimer() {
        CudaCheck(cudaEventCreate(&start_event));
        CudaCheck(cudaEventCreate(&stop_event));
    }

    ~CudaTimer() {
        CudaCheck(cudaEventDestroy(start_event));
        CudaCheck(cudaEventDestroy(stop_event));
    }

    void start(cudaStream_t st = 0) {
        stream = st;
        CudaCheck(cudaEventRecord(start_event, stream));
    }

    float stop() {
        float milliseconds = 0.;
        CudaCheck(cudaEventRecord(stop_event, stream));
        CudaCheck(cudaEventSynchronize(stop_event));
        CudaCheck(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
        return milliseconds;
    }

  private:
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    cudaStream_t stream;
};

}  // namespace tiledcuda
