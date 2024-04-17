#pragma once

#include "cell/copy/dyn_copy.hpp"
#include "cuda_utils.hpp"

using namespace tiledcuda::cell::copy;

namespace tiledcuda::kernels {

/// GEMM kernel using cutlass3 APIs
template <typename Element, typename KeTraits>
__global__ void StaticCuteGemm(const Element* dA, const Element* dB,
                               Element* dC) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<Element*>(shared_buf);

    // Whole GEMM shape
    const int kM = KeTraits::kM;
    const int kN = KeTraits::kN;
    const int kK = KeTraits::kK;

    // CTA GEMM shape
    const int kTM = KeTraits::kTM;
    const int kTN = KeTraits::kTN;
    const int kTK = KeTraits::kTK;

    int tid = threadIdx.x;

    // Advance to the global data tile to the current CTA.
    Element* gA_ptr = const_cast<Element*>(dA) + blockIdx.x * kK * kTM;
    Element* gB_ptr = const_cast<Element*>(dB) + blockIdx.y * kK * kTN;
    Element* gC_ptr = dC + blockIdx.x * kTM * kN + blockIdx.y * kTN;

    // pointers to shared memory tiles
    Element* sA_ptr = shm;
    Element* sB_ptr = shm + kTM * kTK;

    // declare global to shared memory copy plan
    typename KeTraits::LoadA_G2S sA;
    typename KeTraits::LoadB_G2S sB;

    // declare shared memory to register file copy plan.
    // tcu's wmma instruction prescribes a strict data to thread
    // mapping, in the current implementation, the shm-2-reg copy
    // plan is related to mma.
    typename KeTraits::TiledMma mma;
    auto rA = make_s2rA(sA_ptr, tid, typename KeTraits::SmemLayoutA{}, mma);
    auto rB = make_s2rB(sB_ptr, tid, typename KeTraits::SmemLayoutB{}, mma);
    auto acc = get_acc<kTM, kTN>(mma);

    typename KeTraits::StoreC_R2S sC;  // declare register to shared store plan
    typename KeTraits::StoreC_S2G gC;  // declare shm to global store plan

    for (int k = 0; k < kK; k += kTK) {  // iterator over K
        sA.copy(gA_ptr, sA_ptr, tid);
        sB.copy(gB_ptr, sB_ptr, tid);
        __copy_async();
        __syncthreads();

        for (int i = 0; i < rA.get_iters(); ++i) {
            rA.copy(i);  // load A register tile from shared memory
            rB.copy(i);  // load B register tile from shared memory

            gemm(mma, rA[i], rB[i],
                 acc);  // compute using tcu's wmma instruction
        }
        __syncthreads();

        gA_ptr += kTK;
        gB_ptr += kTK;
    }

    sC.copy(acc, shm, tid);  // store register tile to shared memory
    __syncthreads();

    gC.copy(shm, gC_ptr,
            tid);  // store shared memory tile to global memory
}

template <typename Element, typename KeTraits>
__global__ void DynamicCuteGemm(const Element* dA, const Element* dB,
                                Element* dC) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<Element*>(shared_buf);

    // Whole GEMM shape
    const int kM = KeTraits::kM;
    const int kN = KeTraits::kN;
    const int kK = KeTraits::kK;

    // CTA GEMM shape
    const int kTM = KeTraits::kTM;
    const int kTN = KeTraits::kTN;
    const int kTK = KeTraits::kTK;

    int tid = threadIdx.x;

    // Advance to the global data tile to the current CTA.
    Element* gA_ptr = const_cast<Element*>(dA) + blockIdx.x * kK * kTM;
    Element* gB_ptr = const_cast<Element*>(dB) + blockIdx.y * kK * kTN;
    Element* gC_ptr = dC + blockIdx.x * kTM * kN + blockIdx.y * kTN;

    // pointers to shared memory tiles
    Element* sA_ptr = shm;
    Element* sB_ptr = shm + kTM * kTK;

    // declare global to shared memory copy layout.
    auto load_a_g2s_layout = make_row_major_layout(kTM, kTK, kK);
    auto load_b_g2s_layout = make_row_major_layout(kTN, kTK, kK);
    auto store_c_s2g_layout = make_row_major_layout(kTM, kTN, kN);

    // declare shared memory to register file copy plan.
    // tcu's wmma instruction prescribes a strict data to thread
    // mapping, in the current implementation, the shm-2-reg copy
    // plan is related to mma.
    typename KeTraits::TiledMma mma;
    typename KeTraits::TiledCopy tiled_copy;

    auto rA = make_s2rA(sA_ptr, tid, typename KeTraits::SmemLayoutA{}, mma);
    auto rB = make_s2rB(sB_ptr, tid, typename KeTraits::SmemLayoutB{}, mma);
    auto acc = get_acc<kTM, kTN>(mma);

    typename KeTraits::StoreC_R2S sC;  // declare register to shared store plan

    for (int k = 0; k < kK; k += kTK) {  // iterator over K
        copy_tensor_g2s(gA_ptr, sA_ptr, load_a_g2s_layout,
                        typename KeTraits::SmemLayoutA{}, tiled_copy, tid);
        copy_tensor_g2s(gB_ptr, sB_ptr, load_b_g2s_layout,
                        typename KeTraits::SmemLayoutB{}, tiled_copy, tid);
        __copy_async();
        __syncthreads();

        for (int i = 0; i < rA.get_iters(); ++i) {
            rA.copy(i);  // load A register tile from shared memory
            rB.copy(i);  // load B register tile from shared memory

            gemm(mma, rA[i], rB[i],
                 acc);  // compute using tcu's wmma instruction
        }
        __syncthreads();

        gA_ptr += kTK;
        gB_ptr += kTK;
    }

    sC.copy(acc, shm, tid);  // store register tile to shared memory
    __syncthreads();

    copy_tensor_s2g(sC_ptr, gC_ptr, typename KeTraits::SmemLayoutE{},
                    store_c_s2g_layout, tiled_copy,
                    tid);  // store shared memory tile to global memory
}
}  // namespace tiledcuda::kernels