#include "kernels/scatter_nd.hpp"

#include <torch/script.h>

namespace tiledcuda::kernels {
static void custom_scatter_op(torch::Tensor& data, const torch::Tensor& updates,
                              const torch::Tensor& indices) {
    auto dtype = data.dtype();
    if (dtype == torch::kFloat32) {
        scatter_nd<float>(data, updates, indices);
    } else if (dtype == torch::kHalf) {
        // scatter_nd<__half>(data, updates, indices);
    }
}

TORCH_LIBRARY(tiledcuda, t) { t.def("scatter_nd", &custom_scatter_op); }
}  // namespace tiledcuda::kernels