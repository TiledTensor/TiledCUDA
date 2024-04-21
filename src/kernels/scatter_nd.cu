#include "kernels/scatter_nd.hpp"

#include <torch/script.h>

namespace tiledcuda::kernels {
// reference:
// https://github.com/InfiniTensor/RefactorGraph/blob/master/src/04kernel/cuda/src/scatter_nd.cu#L7
// TODO: optimize the kernel by increasing the number of threads to perform
// `atomic_add` operations under `slice_size`.
/**
 * @brief The ScatterNdkernel updates the content of `updates` into `data` based
 * on the index information provided in the given `indices`.
 *
 * @param in The input tensor `updates`.
 * @param out The output tensor `data`.
 * @param indices The indices tensor.
 * @param strides record the stride information between different dimensions in
 * the `data` tensor.
 * @param n The number of indices.
 * @param rank The last dimension of `indices`.
 * @param slice_size The length of the slice to be updated. Specifically, it is
 * the product of the difference between the rank of `data` and the last
 * dimension of `indices` along the memory dimensions of `data`.
 */
template <typename T>
__global__ void scatter_nd_kernel(const T* in, T* out, const int64_t* indices,
                                  unsigned int const* __restrict__ strides,
                                  size_t n, size_t rank, size_t slice_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x,
                step = blockDim.x * gridDim.x;
         tid < n; tid += step) {
        if (tid < n) {
            // tid = indices_index
            unsigned int out_index = 0;
            // the rank of `data`.
            auto i = indices + tid * rank;
            // Compute the offset in the output.
            // j = i[0] * strides[0] + i[1] * strides[1] + ... + i[k] *
            // strides[k]

            for (auto k = 0; k < rank; ++k) {
                out_index += i[k] * __ldg(strides + k);
            };
            for (size_t offset = 0; offset < slice_size; ++offset) {
                atomicAdd(out + out_index + offset,
                          in[tid * slice_size + offset]);
            }
        }
    }
}

template <typename T>
void scatter_nd(torch::Tensor& data, const torch::Tensor& updates,
                const torch::Tensor& indices) {
    auto data_dims = data.sizes();
    auto update_dims = updates.sizes();
    auto indices_dims = indices.sizes();

    // k is the last dimension of indices.
    int64_t k = indices_dims[indices_dims.size() - 1];

    // the rank of data.
    size_t rank = data_dims.size();

    unsigned int* strides = new unsigned int[rank];
    strides[rank - 1] = 1;

    for (int64_t i = rank - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * data_dims[i + 1];
    }

    unsigned int* device_strides;
    CudaCheck(cudaMalloc(&device_strides, rank * sizeof(unsigned int)));
    CudaCheck(cudaMemcpy(device_strides, strides, rank * sizeof(unsigned int),
                         cudaMemcpyHostToDevice));

    // `n` is the product of all dimensions excluding the innermost
    // dimension of `indices`.
    size_t n = indices.numel() / k;

    size_t slice_size = 1;
    for (size_t i = k; i < rank; ++i) {
        slice_size *= data_dims[i];
    }

    size_t data_size = data.numel();

    // #ifdef DEBUG
    //     for (int i = rank - 1; i >= 0; --i) {
    //         std::cout << "strides[" << i << "]: " << strides[i] << std::endl;
    //     }
    //     for (int i = rank - 1; i >= 0; --i) {
    //         std::cout << "data_dims[" << i << "]: " << data_dims[i] <<
    //         std::endl;
    //     }
    //     std::cout << "k: " << k << ", rank: " << rank << std::endl;
    //     std::cout << "n: " << n << ", slice_size: " << slice_size <<
    //     std::endl; std::cout << "data_size: " << data_size << std::endl;
    // #endif

    // TODO: Add some assertion checks.

    int64_t block = 256;
    int64_t grid = (n + block - 1) / block;

    scatter_nd_kernel<<<grid, block>>>(
        reinterpret_cast<const T*>(indices.const_data_ptr()),
        reinterpret_cast<T*>(data.mutable_data_ptr()),
        reinterpret_cast<const int64_t*>(indices.const_data_ptr()),
        reinterpret_cast<const unsigned int*>(device_strides), n, k,
        slice_size);
}

void custom_scatter_op(torch::Tensor& data, const torch::Tensor& updates,
                       const torch::Tensor& indices) {
    auto dtype = data.dtype();
    if (dtype == torch::kFloat32) {
        scatter_nd<float>(data, updates, indices);
    } else if (dtype == torch::kHalf) {
        scatter_nd<__half>(data, updates, indices);
    }
}

}  // namespace tiledcuda::kernels
