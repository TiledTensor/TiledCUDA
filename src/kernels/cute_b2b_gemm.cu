#include "cell/mod.hpp"
#include "kernels/cute_b2b_gemm.hpp"
#include "layout.hpp"

namespace tiledcuda::kernels {

using namespace tiledcuda::cell;
using namespace tiledcuda::cell::copy;

// D = A @ B @ C
template <typename Element, typename KeTraits>
__global__ void dyn_back2back_gemm(const Element* d_a, const Element* d_b,
                                   const Element* d_c, Element* d_d, int m,
                                   int n, int k, int p) {
    // whole problem shape
    // const int kM = KeTraits::kM;
    // const int kN = KeTraits::kN;
    // const int kK = KeTraits::kK;
    // const int kP = KeTraits::kP;

    const int kM = m;
    const int kN = n;
    const int kK = k;
    const int kP = p;

    // shared memory tile shape
    const int kTM = KeTraits::kTM;
    const int kTN = KeTraits::kTN;
    const int kTK = KeTraits::kTK;
    const int kTP = KeTraits::kTP;

    // Advance to the global data tile to the current CTA.
    Element* A = const_cast<Element*>(d_a) + blockIdx.x * (kTM * kK);
    Element* B = const_cast<Element*>(d_b);
    Element* gC_ptr = const_cast<Element*>(d_c) + blockIdx.y * (kTP * kN);
    Element* gD_ptr = d_d + blockIdx.x * (kTM * kP) + (blockIdx.y * kTP);

    Element* gA_ptr;
    Element* gB_ptr;

    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<Element*>(shared_buf);
    // pointers to shared memory tiles
    Element* sA_ptr = shm;
    Element* sB_ptr = shm + kTM * kTK;
    Element* sC_ptr = shm + kTM * kTK + kTK * kTN;
    Element* sD_ptr = shm;

    int tid = threadIdx.x;

    auto load_a_g2s_layout = make_row_major_layout(kTM, kTK, kK);
    auto load_b_g2s_layout = make_row_major_layout(kTN, kTK, kK);
    auto load_c_g2s_layout = make_row_major_layout(kTP, kTN, kN);
    auto store_d_s2g_layout = make_row_major_layout(kTM, kTP, kP);

    typename KeTraits::TiledMma mma;  // for shared memory to register copy
    typename KeTraits::TiledCopy tiled_copy;

    auto rA = make_s2rA(sA_ptr, tid, typename KeTraits::SmemLayoutA{}, mma);
    auto rB = make_s2rB(sB_ptr, tid, typename KeTraits::SmemLayoutB{}, mma);
    auto acc1 = get_acc<kTM, kTN>(mma);  // accumulator for the 1st gemm

    auto rC = make_s2rB(sC_ptr, tid, typename KeTraits::SmemLayoutC{}, mma);
    auto acc2 = get_acc<kTM, kTP>(mma);  // accumulator for the 2nd gemm

    typename KeTraits::StoreD_R2S sD;  // declare register to shared store plan

    for (int n = 0; n < kN; n += kTN) {  // iterate over N
        gA_ptr = A;                      // A tile is repeated loaded
        gB_ptr = B + n * kK;
        for (int k = 0; k < kK; k += kTK) {  // iterate over K

            copy_tensor_g2s(gA_ptr, sA_ptr, load_a_g2s_layout,
                            typename KeTraits::SmemLayoutA{}, tiled_copy, tid);
            copy_tensor_g2s(gB_ptr, sB_ptr, load_b_g2s_layout,
                            typename KeTraits::SmemLayoutB{}, tiled_copy, tid);

            __copy_async();
            __syncthreads();

            // iterate over the register tiles along the kTK
            // dimension
            for (int i = 0; i < rA.get_iters(); ++i) {
                rA.copy(i);  // load A register tile from shared memory
                rB.copy(i);  // load B register tile from shared memory
                gemm(mma, rA[i], rB[i], acc1);  // compute
            }
            __syncthreads();

            gA_ptr += kTK;
            gB_ptr += kTK;
        }
        // The output type of the first tensor core matrix
        // multiplication is float32. However, before the second
        // GEMM operation, the output needs to be converted to half
        // precision.
        auto acc_half = convert_type<Element>(acc1);
        auto rA2 = convert_layout<KeTraits::TiledMma>(acc_half);

        // load C tile from global to shared memory
        // sC.copy(gC_ptr, sC_ptr, tid);
        copy_tensor_g2s(gC_ptr, sC_ptr, load_c_g2s_layout,
                        typename KeTraits::SmemLayoutC{}, tiled_copy, tid);
        __copy_async();
        __syncthreads();

        // iterate over register tiles along the kTN dimension
        for (int i = 0; i < rC.get_iters(); ++i) {
            rC.copy(i);  // load C tile from shared memory to register
            gemm(mma, rA2[i], rC[i], acc2);  // compute
        }
        __syncthreads();

        clear(acc1);
        gC_ptr += kTN;
    }

    sD.copy(acc2, shm,
            tid);  // store register tile to shared memory
    __syncthreads();
    copy_tensor_s2g(sD_ptr, gD_ptr, typename KeTraits::SmemLayoutD{},
                    store_d_s2g_layout, tiled_copy, tid);
}
}  // namespace tiledcuda::kernels