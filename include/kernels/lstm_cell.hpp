#pragma once

#include "cuda_utils.hpp"

#include <torch/script.h>

namespace tiledcuda::kernels {
template <typename Element, typename KeTraits>
__global__ void dyn_lstm_gate(const Element* ws, const Element* us,
                              const Element* xs, const Element* hs, Element* ts,
                              const int m, const int n, const int k);

template <typename Element>
__global__ void lstm_element_wise(const Element* i, const Element* f,
                                  const Element* o, const Element* c0,
                                  const Element* c1, Element* c2, Element* h,
                                  int block_size, int size);

template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape>
void lstm_gate(const Element* w, const Element* x, const Element* u,
               const Element* h, Element* t, const int m, const int n,
               const int k);

template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape>
void lstm_cell(const Element* w, const Element* x, const Element* u,
               const Element* c, const Element* h, Element* c_out,
               Element* h_out, int m, int n, int k);

void custom_lstm_cell_op(const torch::Tensor& w, const torch::Tensor& x,
                         const torch::Tensor& u, const torch::Tensor& c0,
                         const torch::Tensor& h0, torch::Tensor& c1,
                         torch::Tensor& h1, int64_t batch_size,
                         int64_t hidden_size);

}  // namespace tiledcuda::kernels
