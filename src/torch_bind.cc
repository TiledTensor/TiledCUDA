#include "kernels/mod.hpp"

#include <torch/script.h>

namespace tiledcuda {
using namespace tiledcuda::kernels;

TORCH_LIBRARY(tiledcuda, t) {
    t.def("scatter_nd", &custom_scatter_op);
    t.def("gemm", &custom_gemm_op);
    t.def("back2back_gemm", &custom_back2back_op);
};

}  // namespace tiledcuda