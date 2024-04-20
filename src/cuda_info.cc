#include "cuda_info.hpp"

#include "cuda_utils.hpp"

namespace tiledcuda {
int GetGPUMaxThreadsPerMultiProcessor(int id) {
    int count;
    CudaCheck(cudaDeviceGetAttribute(
        &count, cudaDevAttrMaxThreadsPerMultiProcessor, id));
    return count;
}
}  // namespace tiledcuda