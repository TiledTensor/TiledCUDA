#pragma once

#include "cuda_utils.hpp"

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

}  // namespace tiledcuda::kernels