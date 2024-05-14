#include "cell/mod.hpp"
#include "kernels/gemm.hpp"

namespace tiledcuda::kernels {

using namespace tiledcuda::cell;
using namespace tiledcuda::cell::copy;
using namespace tiledcuda::cell::compute;
namespace tl = tiledcuda::tile_layout;

template <typename Element, typename KeTraits>
__global__ void dyn_cute_gemm_kernel(const Element* dA, const Element* dB,
                                     Element* dC, int m, int n, int k) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<Element*>(shared_buf);

    // Whole GEMM shape
    const int kN = n;
    const int kK = k;

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
    Element* sC_ptr = shm;

    // declare global to shared memory copy layout.
    auto load_a_g2s_layout = tl::make_row_major_layout(kTM, kTK, kK);
    auto load_b_g2s_layout = tl::make_row_major_layout(kTN, kTK, kK);
    auto store_c_s2g_layout = tl::make_row_major_layout(kTM, kTN, kN);

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
        copy_2d_tile_g2s(gA_ptr, sA_ptr, load_a_g2s_layout,
                         typename KeTraits::SmemLayoutA{}, tiled_copy, tid);
        copy_2d_tile_g2s(gB_ptr, sB_ptr, load_b_g2s_layout,
                         typename KeTraits::SmemLayoutB{}, tiled_copy, tid);
        __copy_async();
        __syncthreads();

        for (int i = 0; i < rA.get_iters(); ++i) {
            rA.copy(i);  // load A register tile from shared memory
            rB.copy(i);  // load B register tile from shared memory

            // compute using tcu's wmma instruction
            gemm(mma, rA[i], rB[i], acc);
        }
        __syncthreads();

        gA_ptr += kTK;
        gB_ptr += kTK;
    }

    sC.copy(acc, shm, tid);  // store register tile to shared memory
    __syncthreads();

    copy_2d_tile_s2g(sC_ptr, gC_ptr, typename KeTraits::SmemLayoutC{},
                     store_c_s2g_layout, tiled_copy,
                     tid);  // store shared memory tile to global memory
}

template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape>
void cute_gemm(const Element* a, const Element* b, Element* c, int m, int n,
               int k) {
    // CTA GEMM shape
    static const int kTM = dim_size<0, CtaTileShape>;
    static const int kTN = dim_size<1, CtaTileShape>;
    static const int kTK = dim_size<2, CtaTileShape>;

    static_assert(kTM % dim_size<0, WarpArrangement> == 0,
                  "the M dimension of the CTA tile should be "
                  "divisible by the "
                  "number of warps along that that dimension.");
    static_assert(kTN % dim_size<1, WarpArrangement> == 0,
                  "the N dimension of the CTA tile should be "
                  "divisible by the "
                  "number of warps along that that dimension.");

    using GemmTraits =
        traits::DynGemmTraits<Element, InstructionShape, ValueMnk,
                              WarpArrangement, CtaTileShape>;

    static constexpr int smem_size =
        std::max(kTK * (kTN + kTM), kTM * kTN) * sizeof(Element);

    auto gemm_kernel = &dyn_cute_gemm_kernel<Element, GemmTraits>;

    // maximal statically allocated smem per block
    const int kMaxSmemPerBlock = 48 * 1024;
    if (smem_size > kMaxSmemPerBlock) {
        cudaFuncSetAttribute(gemm_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
    }

    const int block_m = (m + kTM - 1) / kTM;
    const int block_n = (n + kTN - 1) / kTN;

    const int kThreads = GemmTraits::kThreads;

    dim3 gridDim(block_m, block_n);
    dim3 blockDim(kThreads, 1, 1);

    gemm_kernel<<<gridDim, blockDim, smem_size>>>(a, b, c, m, n, k);
}

void custom_gemm_op(const torch::Tensor& a, const torch::Tensor& b,
                    torch::Tensor& c, int64_t m, int64_t n, int64_t k) {
    using InstructionShape = cell::TileShape<16, 8, 16>;
    using ValueMnk = cell::TileShape<1, 2, 1>;
    using WarpArrangement = cell::TileShape<1, 1, 1>;
    using CtaTileShape = cell::TileShape<16, 32, 32>;

    auto dtype = a.dtype();
    if (dtype == torch::kFloat32) {
        // TODO: Add support for fp32.
    } else if (dtype == torch::kHalf) {
        cute_gemm<cutlass::half_t, InstructionShape, ValueMnk, WarpArrangement,
                  CtaTileShape>(
            reinterpret_cast<const cutlass::half_t*>(a.const_data_ptr()),
            reinterpret_cast<const cutlass::half_t*>(b.const_data_ptr()),
            reinterpret_cast<cutlass::half_t*>(c.mutable_data_ptr()), m, n, k);
    }
}

}  // namespace tiledcuda::kernels
