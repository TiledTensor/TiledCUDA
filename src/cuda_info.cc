#include "cuda_info.hpp"

#include <sstream>
#include <vector>

namespace tiledcuda {
// Returns the number of GPUs.
int GetGPUDeviceCount() {
    int deviceCount = 0;
    CudaCheck(cudaGetDeviceCount(&deviceCount));
    return deviceCount;
}

// Returns the compute capability of the given GPU.
int GetGPUComputeCapability(int id) {
    int major, minor;
    CudaCheck(
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, id));
    CudaCheck(
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, id));
    return major * 10 + minor;
}

// Returns the number of multiprocessors for the given GPU.
int GetGPUMultiProcessors(int id) {
    int count;
    CudaCheck(
        cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, id));
    return count;
}

// Returns the maximum number of threads per multiprocessor for the given GPU.
int GetGPUMaxThreadsPerMultiProcessor(int id) {
    int count;
    CudaCheck(cudaDeviceGetAttribute(
        &count, cudaDevAttrMaxThreadsPerMultiProcessor, id));
    return count;
}

// Returns the maximum number of threads per block for the given GPU.
int GetGPUMaxThreadsPerBlock(int id) {
    int count;
    CudaCheck(
        cudaDeviceGetAttribute(&count, cudaDevAttrMaxThreadsPerBlock, id));
    return count;
}

// Returns the maximum grid size for the given GPU.
dim3 GetGpuMaxGridDimSize(int id) {
    dim3 grid_size;

    int size;
    CudaCheck(cudaDeviceGetAttribute(&size, cudaDevAttrMaxGridDimX, id));
    grid_size.x = size;

    CudaCheck(cudaDeviceGetAttribute(&size, cudaDevAttrMaxGridDimY, id));
    grid_size.y = size;

    CudaCheck(cudaDeviceGetAttribute(&size, cudaDevAttrMaxGridDimZ, id));
    grid_size.z = size;
    return grid_size;
}

// Returns the name of the device.
std::string GetDeviceName() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::stringstream ss(prop.name);
    const char delim = ' ';

    std::string s;
    std::vector<std::string> out;

    while (std::getline(ss, s, delim)) {
        out.push_back(s);
    }

    std::stringstream out_ss;
    int i = 0;
    for (; i < static_cast<int>(out.size()) - 1; ++i) out_ss << out[i] << "_";
    out_ss << out[i];
    return out_ss.str();
}
}  // namespace tiledcuda
