#include "cell/mod.hpp"
#include "cuda_info.hpp"
#include "errors.hpp"
#include "kernels/lstm_cell.hpp"
#include "types/layout.hpp"

namespace tiledcuda::kernels {

using namespace tiledcuda::cell;
using namespace tiledcuda::cell::copy;
using namespace tiledcuda::cell::compute;

namespace tl = tiledcuda::cell::tile_layout;

template <typename Element, typename KeTraits>
__global__ void dyn_lstm_gate(const Element* ws, const Element* us,
                              const Element* xs, const Element* hs, Element* ts,
                              const int m, const int n, const int k) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<Element*>(shared_buf);

    const int kN = n;
    const int kK = k;

    // CTA GEMM shape
    const int kTM = KeTraits::kTM;
    const int kTN = KeTraits::kTN;
    const int kTK = KeTraits::kTK;

    // Advance to the global data tile to the current CTA.
    Element* gxs_ptr = const_cast<Element*>(xs) + blockIdx.y * kK * kTN;
    Element* ghs_ptr = const_cast<Element*>(hs) + blockIdx.y * kK * kTN;
    Element* gws_ptr = const_cast<Element*>(ws) + blockIdx.x * kK * kTM;
    Element* gus_ptr = const_cast<Element*>(us) + blockIdx.x * kK * kTM;

    Element* gts_ptr = ts + blockIdx.x * kTM * kN + blockIdx.y * kTN;

    int total_block_x = gridDim.x;
    int current_block_x = blockIdx.x;

    // pointers to shared memory tiles
    Element* sws_ptr = shm;
    Element* sxs_ptr = shm + kTM * kTK;
    Element* sus_ptr = shm + kTM * kTK + kTK * kTN;
    Element* shs_ptr = shm + kTM * kTK + kTK * kTN + kTM * kTK;
    Element* sts_ptr = shm;

    // declare shared memory to register file copy plan.
    // tcu's wmma instruction prescribes a strict data to thread
    // mapping, in the current implementation, the shm-2-reg copy
    // plan is related to mma.
    typename KeTraits::TiledMma mma;
    typename KeTraits::TiledCopyG2S tiled_copy;

    auto rws = make_s2rA(sws_ptr, typename KeTraits::SmemLayoutA{}, mma);
    auto rxs = make_s2rB(sxs_ptr, typename KeTraits::SmemLayoutB{}, mma);
    auto rus = make_s2rA(sus_ptr, typename KeTraits::SmemLayoutC{}, mma);
    auto rhs = make_s2rB(shs_ptr, typename KeTraits::SmemLayoutD{}, mma);

    auto acc1 = get_acc<kTM, kTN>(mma);
    auto acc2 = get_acc<kTM, kTN>(mma);

    auto load_a_g2s_layout = tl::make_row_major_layout(kTM, kTK, kK);
    auto load_b_g2s_layout = tl::make_row_major_layout(kTN, kTK, kK);
    auto load_c_g2s_layout = tl::make_row_major_layout(kTM, kTK, kK);
    auto load_d_g2s_layout = tl::make_row_major_layout(kTN, kTK, kK);
    auto store_e_s2g_layout = tl::make_row_major_layout(kTM, kTN, kN);

    typename KeTraits::StoreE_R2S sts;  // declare register to shared store

    for (int k = 0; k < kK; k += kTK) {
        // TODO: Load data from global memory to shared memory
        copy_2d_tile_g2s(gws_ptr, sws_ptr, load_a_g2s_layout,
                         typename KeTraits::SmemLayoutA{}, tiled_copy);
        copy_2d_tile_g2s(gxs_ptr, sxs_ptr, load_b_g2s_layout,
                         typename KeTraits::SmemLayoutB{}, tiled_copy);
        copy_2d_tile_g2s(gus_ptr, sus_ptr, load_c_g2s_layout,
                         typename KeTraits::SmemLayoutC{}, tiled_copy);
        copy_2d_tile_g2s(ghs_ptr, shs_ptr, load_d_g2s_layout,
                         typename KeTraits::SmemLayoutD{}, tiled_copy);

        __copy_async();
        __syncthreads();

        for (int i = 0; i < rws.get_iters(); i++) {
            rws.copy(i);
            rxs.copy(i);
            gemm(mma, rws[i], rxs[i], acc1);
        }

        for (int i = 0; i < rus.get_iters(); i++) {
            rus.copy(i);
            rhs.copy(i);
            gemm(mma, rus[i], rhs[i], acc2);
        }

        __syncthreads();
        gws_ptr += kTK;
        gxs_ptr += kTK;
        gus_ptr += kTK;
        ghs_ptr += kTK;
    }

    __syncthreads();
    cute::axpby(1.0, acc1, 1.0, acc2);

    __syncthreads();
    if (current_block_x < total_block_x * 3 / 4) {
        cute_sigmoid(acc2);
    } else {
        cute_tanh(acc2);
    }
    __syncthreads();

    sts.copy(acc2, shm);

    __syncthreads();

    copy_2d_tile_s2g(sts_ptr, gts_ptr, typename KeTraits::SmemLayoutE{},
                     store_e_s2g_layout, typename KeTraits::TiledCopyS2G{});
}

template <typename Element>
__global__ void lstm_element_wise(const Element* i, const Element* f,
                                  const Element* o, const Element* c_candidate,
                                  const Element* c, Element* c_out,
                                  Element* h_out, int block_size, int size) {
    int index = blockIdx.x * block_size + threadIdx.x;

    if (index < size) {
        // TODO: Loading data into shared memory and computing, versus
        // computing directly in global memory, does not seem to make a
        // difference. This seems to require further optimization, such as
        // reconsidering redistributing data to different threads and performing
        // vectorized loading and storing.

        // This is a very naive kernel that loads data into shared memory and
        // then performs computations. It has been temporarily commented out.

        c_out[index] = f[index] * c[index] + i[index] * c_candidate[index];

        __syncthreads();

        h_out[index] = o[index] * tanh(c_out[index]);
    }
}

template <typename Element, typename CtaTileShape>
void lstm_gate(const Element* w, const Element* x, const Element* u,
               const Element* h, Element* t, const int m, const int n,
               const int k) {
    // Whole GEMM shape
    static const int kM = m;
    static const int kN = n;

    // CTA GEMM shape
    static const int kTM = dim_size<0, CtaTileShape>;
    static const int kTN = dim_size<1, CtaTileShape>;
    static const int kTK = dim_size<2, CtaTileShape>;

    using KeTraits = traits::DynLstmGateTraits<Element, CtaTileShape>;

    static constexpr int smem_size =
        std::max(kTK * (kTN + kTM) * 2, kTM * kTN) * sizeof(Element);

    auto lstm_gate = &dyn_lstm_gate<Element, KeTraits>;

    // maximal statically allocated smem per block
    const int kMaxSmemPerBlock = 48 * 1024;
    if (smem_size > kMaxSmemPerBlock) {
        cudaFuncSetAttribute(
            lstm_gate, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    const int block_m = (kM + kTM - 1) / kTM;
    const int block_n = (kN + kTN - 1) / kTN;

#ifdef DEBUG
    std::cout << "block_m: " << block_m << ", block_n: " << block_n
              << std::endl;
#endif

    const int kThreads = KeTraits::kThreads;

    dim3 gridDim(block_m, block_n, 1);
    dim3 blockDim(kThreads, 1, 1);

    lstm_gate<<<gridDim, blockDim, smem_size>>>(w, u, x, h, t, m, n, k);
}

template <typename Element, typename CtaTileShape>
void lstm_cell(const Element* w, const Element* x, const Element* u,
               const Element* c, const Element* h, Element* c_out,
               Element* h_out, int m, int n, int k) {
    static const int kM = m;
    static const int kN = n;

    static const int M = kM / 4;
    static const int N = kN;

    // Cuda malloc for output
    Element* t;
    CudaCheck(cudaMalloc(&t, m * n * sizeof(Element)));

    lstm_gate<Element, CtaTileShape>(w, x, u, h, t, m, n, k);

    const Element* i = t;
    const Element* f = t + M * N;
    const Element* o = t + 2 * M * N;
    const Element* c_candidate = t + 3 * M * N;

    auto element_wise = &lstm_element_wise<Element>;

    /*
    TODO: Use `kMaxThreads` will case a runtime error:
    ```
    RuntimeError: CUDA error: invalid configuration argument
    CUDA kernel errors might be asynchronously reported at some other API call,
    so the stacktrace below might be incorrect. For debugging consider passing
    CUDA_LAUNCH_BLOCKING=1. Compile with `TORCH_USE_CUDA_DSA` to enable
    device-side assertions.
    ```
    */
    // int kMaxThreads = GetGPUMaxThreadsPerMultiProcessor(0);
    int size = M * N;
    int block_threads = 512;
    int block_size = (size + block_threads - 1) / block_threads;
    dim3 element_wise_grid_dim(block_size, 1, 1);
    dim3 element_wise_block_dim(block_threads, 1, 1);

    element_wise<<<element_wise_grid_dim, element_wise_block_dim>>>(
        i, f, o, c_candidate, c, c_out, h_out, block_threads, size);

    CudaCheck(cudaFree(t));
}

void custom_lstm_cell_op(const torch::Tensor& w, const torch::Tensor& x,
                         const torch::Tensor& u, const torch::Tensor& c0,
                         const torch::Tensor& h0, torch::Tensor& c1,
                         torch::Tensor& h1, int64_t batch_size,
                         int64_t hidden_size) {
    using CtaTileShape = cell::TileShape<16, 32, 32>;

    auto dtype = w.dtype();

    int m = 4 * hidden_size;
    int n = batch_size;
    int k = hidden_size;

    if (dtype == torch::kHalf) {
        lstm_cell<cutlass::half_t, CtaTileShape>(
            reinterpret_cast<const cutlass::half_t*>(w.const_data_ptr()),
            reinterpret_cast<const cutlass::half_t*>(x.const_data_ptr()),
            reinterpret_cast<const cutlass::half_t*>(u.const_data_ptr()),
            reinterpret_cast<const cutlass::half_t*>(c0.const_data_ptr()),
            reinterpret_cast<const cutlass::half_t*>(h0.const_data_ptr()),
            reinterpret_cast<cutlass::half_t*>(c1.mutable_data_ptr()),
            reinterpret_cast<cutlass::half_t*>(h1.mutable_data_ptr()), m, n, k);
    } else {
        throw NotImplementedException("Unsupported data type.");
    }
}
}  // namespace tiledcuda::kernels
