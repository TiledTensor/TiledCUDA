#include "cell/mod.hpp"
#include "errors.hpp"
#include "kernels/bmm.hpp"

namespace tiledcuda::kernels {

using namespace tiledcuda::cell;
using namespace tiledcuda::cell::copy;
using namespace tiledcuda::cell::compute;

namespace tl = tiledcuda::cell::tile_layout;

template <typename Element, typename KeTraits>
__global__ void dyn_cute_batched_gemm_kernel(const Element* dA,
                                             const Element* dB, Element* dC,
                                             int m, int n, int k) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<Element*>(shared_buf);

    // Whole GEMM shape
    const int kM = m;
    const int kN = n;
    const int kK = k;

    // CTA GEMM shape
    const int kTM = KeTraits::kTM;
    const int kTN = KeTraits::kTN;
    const int kTK = KeTraits::kTK;

    // Advance to the global data tile to the current CTA.
    Element* gA_ptr =
        const_cast<Element*>(dA) + blockIdx.x * kK * kTM + blockIdx.z * kK * kM;
    Element* gB_ptr =
        const_cast<Element*>(dB) + blockIdx.y * kK * kTN + blockIdx.z * kK * kN;
    Element* gC_ptr =
        dC + blockIdx.x * kTM * kN + blockIdx.y * kTN + blockIdx.z * kM * kN;

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

    auto rA = make_s2rA(sA_ptr, typename KeTraits::SmemLayoutA{}, mma);
    auto rB = make_s2rB(sB_ptr, typename KeTraits::SmemLayoutB{}, mma);
    auto acc = get_acc<kTM, kTN>(mma);

    typename KeTraits::StoreC_R2S sC;  // declare register to shared store plan

    for (int k = 0; k < kK; k += kTK) {  // iterator over K
        copy_2d_tile_g2s(gA_ptr, sA_ptr, load_a_g2s_layout,
                         typename KeTraits::SmemLayoutA{},
                         typename KeTraits::TiledCopyG2S{});
        copy_2d_tile_g2s(gB_ptr, sB_ptr, load_b_g2s_layout,
                         typename KeTraits::SmemLayoutB{},
                         typename KeTraits::TiledCopyG2S{});
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

    sC.copy(acc, shm);  // store register tile to shared memory
    __syncthreads();

    // store shared memory tile to global memory
    copy_2d_tile_s2g(sC_ptr, gC_ptr, typename KeTraits::SmemLayoutC{},
                     store_c_s2g_layout, typename KeTraits::TiledCopyS2G{});
}

template <typename Element, typename CtaTileShape>
void cute_batched_gemm(const Element* a, const Element* b, Element* c, int m,
                       int n, int k, int batch_count) {
    // CTA GEMM shape
    static const int kTM = dim_size<0, CtaTileShape>;
    static const int kTN = dim_size<1, CtaTileShape>;
    static const int kTK = dim_size<2, CtaTileShape>;

    using GemmTraits = traits::DynBatchedGemmTraits<Element, CtaTileShape>;

    static constexpr int smem_size =
        std::max(kTK * (kTN + kTM), kTM * kTN) * sizeof(Element);

    auto batched_gemm_kernel =
        &dyn_cute_batched_gemm_kernel<Element, GemmTraits>;

    // maximal statically allocated smem per block
    const int kMaxSmemPerBlock = 48 * 1024;
    if (smem_size > kMaxSmemPerBlock) {
        cudaFuncSetAttribute(batched_gemm_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
    }

    const int block_m = (m + kTM - 1) / kTM;
    const int block_n = (n + kTN - 1) / kTN;

    const int kThreads = GemmTraits::kThreads;

    dim3 gridDim(block_m, block_n, batch_count);
    dim3 blockDim(kThreads, 1, 1);

    batched_gemm_kernel<<<gridDim, blockDim, smem_size>>>(a, b, c, m, n, k);
}

void custom_batched_gemm_op(const torch::Tensor& a, const torch::Tensor& b,
                            torch::Tensor& c, int64_t m, int64_t n, int64_t k,
                            int64_t batch_count) {
    using CtaTileShape = cell::TileShape<32, 32, 32>;

    auto dtype = a.dtype();

    if (dtype == torch::kHalf) {
        cute_batched_gemm<cutlass::half_t, CtaTileShape>(
            reinterpret_cast<const cutlass::half_t*>(a.const_data_ptr()),
            reinterpret_cast<const cutlass::half_t*>(b.const_data_ptr()),
            reinterpret_cast<cutlass::half_t*>(c.mutable_data_ptr()), m, n, k,
            batch_count);
    } else {
        throw errors::NotImplementedException("Unsupported data type.");
    }
}
}  // namespace tiledcuda::kernels
