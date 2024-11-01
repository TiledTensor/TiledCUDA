// #include "cell/mod.hpp"
// #include "kernels/b2b_gemm.hpp"

// namespace tiledcuda::kernels {

// using namespace tiledcuda::cell;
// using namespace tiledcuda::cell::copy;
// using namespace tiledcuda::cell::compute;

// namespace tl = tiledcuda::cell::tile_layout;

// // D = A @ B @ C
// template <typename Element, typename KeTraits>
// __global__ void dyn_back2back_gemm(const Element* d_a, const Element* d_b,
//                                    const Element* d_c, Element* d_d, int m,
//                                    int n, int k, int p) {
//     // whole problem shape
//     const int kN = n;
//     const int kK = k;
//     const int kP = p;

//     // shared memory tile shape
//     const int kTM = KeTraits::kTM;
//     const int kTN = KeTraits::kTN;
//     const int kTK = KeTraits::kTK;
//     const int kTP = KeTraits::kTP;

//     // Advance to the global data tile to the current CTA.
//     Element* A = const_cast<Element*>(d_a) + blockIdx.x * (kTM * kK);
//     Element* B = const_cast<Element*>(d_b);
//     Element* gC_ptr = const_cast<Element*>(d_c) + blockIdx.y * (kTP * kN);
//     Element* gD_ptr = d_d + blockIdx.x * (kTM * kP) + (blockIdx.y * kTP);

//     Element* gA_ptr;
//     Element* gB_ptr;

//     extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
//     auto* shm = reinterpret_cast<Element*>(shared_buf);
//     // pointers to shared memory tiles
//     Element* sA_ptr = shm;
//     Element* sB_ptr = shm + kTM * kTK;
//     Element* sC_ptr = shm + kTM * kTK + kTK * kTN;
//     Element* sD_ptr = shm;

//     auto load_a_g2s_layout = tl::make_row_major_layout(kTM, kTK, kK);
//     auto load_b_g2s_layout = tl::make_row_major_layout(kTN, kTK, kK);
//     auto load_c_g2s_layout = tl::make_row_major_layout(kTP, kTN, kN);
//     auto store_d_s2g_layout = tl::make_row_major_layout(kTM, kTP, kP);

//     typename KeTraits::TiledMma mma;  // for shared memory to register copy
//     typename KeTraits::TiledCopyG2S tiled_copy;

//     auto rA = make_s2rA(sA_ptr, typename KeTraits::SmemLayoutA{}, mma);
//     auto rB = make_s2rB(sB_ptr, typename KeTraits::SmemLayoutB{}, mma);
//     auto acc1 = get_acc<kTM, kTN>(mma);  // accumulator for the 1st gemm

//     auto rC = make_s2rB(sC_ptr, typename KeTraits::SmemLayoutC{}, mma);
//     auto acc2 = get_acc<kTM, kTP>(mma);  // accumulator for the 2nd gemm

//     typename KeTraits::StoreD_R2S sD;  // declare register to shared store
//     plan

//     for (int n = 0; n < kN; n += kTN) {  // iterate over N
//         gA_ptr = A;                      // A tile is repeated loaded
//         gB_ptr = B + n * kK;
//         for (int k = 0; k < kK; k += kTK) {  // iterate over K
//             copy_2d_tile_g2s(gA_ptr, sA_ptr, load_a_g2s_layout,
//                              typename KeTraits::SmemLayoutA{}, tiled_copy);
//             copy_2d_tile_g2s(gB_ptr, sB_ptr, load_b_g2s_layout,
//                              typename KeTraits::SmemLayoutB{}, tiled_copy);
//             __copy_async();
//             __syncthreads();

//             // iterate over the register tiles along the kTK dimension
//             for (int i = 0; i < rA.get_iters(); ++i) {
//                 rA.copy(i);  // load A register tile from shared memory
//                 rB.copy(i);  // load B register tile from shared memory
//                 gemm(mma, rA[i], rB[i], acc1);  // compute
//             }
//             __syncthreads();

//             gA_ptr += kTK;
//             gB_ptr += kTK;
//         }
//         // The output type of the first tensor core matrix multiplication is
//         // float32. However, before the second GEMM operation, the output
//         // needs to be converted to half precision.
//         auto acc_half = convert_type<Element>(acc1);
//         auto rA2 = convert_layout<KeTraits::TiledMma>(acc_half);

//         // load C tile from global to shared memory
//         copy_2d_tile_g2s(gC_ptr, sC_ptr, load_c_g2s_layout,
//                          typename KeTraits::SmemLayoutC{},
//                          typename KeTraits::TiledCopyS2G{});
//         __copy_async();
//         __syncthreads();

//         // iterate over register tiles along the kTN dimension
//         for (int i = 0; i < rC.get_iters(); ++i) {
//             rC.copy(i);  // load C tile from shared memory to register
//             gemm(mma, rA2[i], rC[i], acc2);  // compute
//         }
//         __syncthreads();

//         clear(acc1);
//         gC_ptr += kTN;
//     }

//     // store register tile to shared memory
//     sD.copy(acc2, shm);
//     __syncthreads();
//     copy_2d_tile_s2g(sD_ptr, gD_ptr, typename KeTraits::SmemLayoutD{},
//                      store_d_s2g_layout, typename KeTraits::TiledCopyS2G{});
// }

// template <typename Element, typename CtaTileShape>
// void cute_back2back_gemm(const Element* d_a, const Element* d_b,
//                          const Element* d_c, Element* d_d, int m, int n, int
//                          k, int p) {
//     static const int kTM = dim_size<0, CtaTileShape>;
//     static const int kTN = dim_size<1, CtaTileShape>;
//     static const int kTK = dim_size<2, CtaTileShape>;
//     static const int kTP = dim_size<3, CtaTileShape>;

//     using KeTraits =
//         cell::traits::DynBack2BackGemmTraits<Element, CtaTileShape>;

//     int shm_input = (kTM * kTK + kTK * kTN + kTN * kTP);
//     int shm_output = kTM * kTP;
//     int shm_size = shm_input < shm_output ? shm_output * sizeof(Element)
//                                           : shm_input * sizeof(Element);

//     auto kernel = &dyn_back2back_gemm<Element, KeTraits>;
//     if (shm_size > 48 * 1024) {
//         cudaFuncSetAttribute(
//             kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
//     }

//     // blocks are launched along the M and P dimensions.
//     int block_x = (m + kTM - 1) / kTM;
//     int block_y = (p + kTP - 1) / kTP;

//     kernel<<<dim3(block_x, block_y, 1), dim3(KeTraits::kThreads, 1, 1),
//              shm_size, 0>>>(d_a, d_b, d_c, d_d, m, n, k, p);
// }

// void custom_back2back_op(const torch::Tensor& a, const torch::Tensor& b,
//                          const torch::Tensor& c, torch::Tensor& d, int64_t m,
//                          int64_t n, int64_t k, int64_t p) {
//     using CtaTileShape = cell::TileShape<32, 32, 32, 32>;

//     auto dtype = a.dtype();
//     if (dtype == torch::kFloat32) {
//         // TODO: Add support for fp32.
//     } else if (dtype == torch::kHalf) {
//         const cutlass::half_t* a_ptr =
//             reinterpret_cast<const cutlass::half_t*>(a.const_data_ptr());
//         const cutlass::half_t* b_ptr =
//             reinterpret_cast<const cutlass::half_t*>(b.const_data_ptr());
//         const cutlass::half_t* c_ptr =
//             reinterpret_cast<const cutlass::half_t*>(c.const_data_ptr());
//         cutlass::half_t* d_ptr =
//             reinterpret_cast<cutlass::half_t*>(d.mutable_data_ptr());

//         cute_back2back_gemm<cutlass::half_t, CtaTileShape>(a_ptr, b_ptr,
//         c_ptr,
//                                                            d_ptr, m, n, k,
//                                                            p);
//     }
// }
// }  // namespace tiledcuda::kernels
