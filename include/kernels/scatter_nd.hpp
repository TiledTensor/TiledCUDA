#pragma once

#include "cuda_utils.hpp"

#include <torch/script.h>

#include <cstdint>

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
__global__ void ScatterNdKernel(const T* in, T* out, const int64_t* indices,
                                unsigned int const* __restrict__ strides,
                                size_t n, size_t rank, size_t slice_size);

template <typename T>
void scatter_nd(torch::Tensor& data, const torch::Tensor& updates,
                const torch::Tensor& indices);

void custom_scatter_op(torch::Tensor& data, const torch::Tensor& updates,
                       const torch::Tensor& indices);

}  // namespace tiledcuda::kernels