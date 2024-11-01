#include "cell/mod.hpp"
#include "kernels/gemm.hpp"
#include "types/mod.hpp"

namespace tiledcuda::kernels {

using namespace tiledcuda;
using namespace tiledcuda::cell;
using namespace tiledcuda::cell::copy;

namespace tl = tile_layout;

// template <typename Element, typename KeTraits>
// __global__ void dyn_cute_gemm_kernel(const Element* dA, const Element* dB,
//                                      Element* dC, int m, int n, int k) {
//     extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
//     auto* shm = reinterpret_cast<Element*>(shared_buf);

//     // Whole GEMM shape
//     const int kN = n;
//     const int kK = k;

//     // CTA GEMM shape
//     const int kTM = KeTraits::kTM;
//     const int kTN = KeTraits::kTN;
//     const int kTK = KeTraits::kTK;

//     // Advance to the global data tile to the current CTA.
//     Element* gA_ptr = const_cast<Element*>(dA) + blockIdx.x * kK * kTM;
//     Element* gB_ptr = const_cast<Element*>(dB) + blockIdx.y * kK * kTN;
//     Element* gC_ptr = dC + blockIdx.x * kTM * kN + blockIdx.y * kTN;

//     // pointers to shared memory tiles
//     Element* sA_ptr = shm;
//     Element* sB_ptr = shm + kTM * kTK;
//     Element* sC_ptr = shm;

//     // declare global to shared memory copy layout.
//     auto load_a_g2s_layout = tl::make_row_major_layout(kTM, kTK, kK);
//     auto load_b_g2s_layout = tl::make_row_major_layout(kTN, kTK, kK);
//     auto store_c_s2g_layout = tl::make_row_major_layout(kTM, kTN, kN);

//     // declare shared memory to register file copy plan.
//     // tcu's wmma instruction prescribes a strict data to thread
//     // mapping, in the current implementation, the shm-2-reg copy
//     // plan is related to mma.
//     typename KeTraits::TiledMma mma;
//     typename KeTraits::TiledCopyG2S tiled_copy;

//     auto rA = make_s2rA(sA_ptr, typename KeTraits::SmemLayoutA{}, mma);
//     auto rB = make_s2rB(sB_ptr, typename KeTraits::SmemLayoutB{}, mma);
//     auto acc = get_acc<kTM, kTN>(mma);

//     typename KeTraits::StoreC_R2S sC;  // declare register to shared store
//     plan

//     for (int k = 0; k < kK; k += kTK) {  // iterator over K
//         copy_2d_tile_g2s(gA_ptr, sA_ptr, load_a_g2s_layout,
//                          typename KeTraits::SmemLayoutA{}, tiled_copy);
//         copy_2d_tile_g2s(gB_ptr, sB_ptr, load_b_g2s_layout,
//                          typename KeTraits::SmemLayoutB{}, tiled_copy);
//         __copy_async();
//         __syncthreads();

//         for (int i = 0; i < rA.get_iters(); ++i) {
//             rA.copy(i);  // load A register tile from shared memory
//             rB.copy(i);  // load B register tile from shared memory

//             // compute using tcu's wmma instruction
//             gemm(mma, rA[i], rB[i], acc);
//         }
//         __syncthreads();

//         gA_ptr += kTK;
//         gB_ptr += kTK;
//     }

//     sC.copy(acc, shm);  // store register tile to shared memory
//     __syncthreads();

//     // store shared memory tile to global memory
//     copy_2d_tile_s2g(sC_ptr, gC_ptr, typename KeTraits::SmemLayoutC{},
//                      store_c_s2g_layout, typename KeTraits::TiledCopyS2G{});
// }

// template <typename Element, typename CtaTileShape>
// void cute_gemm(const Element* a, const Element* b, Element* c, int m, int n,
//                int k) {
//     // CTA GEMM shape
//     static const int kTM = dim_size<0, CtaTileShape>;
//     static const int kTN = dim_size<1, CtaTileShape>;
//     static const int kTK = dim_size<2, CtaTileShape>;

//     using GemmTraits = traits::DynGemmTraits<Element, CtaTileShape>;

//     static constexpr int smem_size =
//         std::max(kTK * (kTN + kTM), kTM * kTN) * sizeof(Element);

//     auto gemm_kernel = &dyn_cute_gemm_kernel<Element, GemmTraits>;

//     // maximal statically allocated smem per block
//     const int kMaxSmemPerBlock = 48 * 1024;
//     if (smem_size > kMaxSmemPerBlock) {
//         cudaFuncSetAttribute(gemm_kernel,
//                              cudaFuncAttributeMaxDynamicSharedMemorySize,
//                              smem_size);
//     }

//     const int block_m = (m + kTM - 1) / kTM;
//     const int block_n = (n + kTN - 1) / kTN;

//     const int kThreads = GemmTraits::kThreads;

//     dim3 gridDim(block_m, block_n);
//     dim3 blockDim(kThreads, 1, 1);

//     gemm_kernel<<<gridDim, blockDim, smem_size>>>(a, b, c, m, n, k);
// }

// void custom_gemm_op(const torch::Tensor& a, const torch::Tensor& b,
//                     torch::Tensor& c, int64_t m, int64_t n, int64_t k) {
//     // shared memory tile shape
//     using CtaTileShape = cell::TileShape<64, 128, 64>;

//     auto dtype = a.dtype();
//     if (dtype == torch::kFloat32) {
//         // TODO: Add support for fp32.
//     } else if (dtype == torch::kHalf) {
//         cute_gemm<cutlass::half_t, CtaTileShape>(
//             reinterpret_cast<const cutlass::half_t*>(a.const_data_ptr()),
//             reinterpret_cast<const cutlass::half_t*>(b.const_data_ptr()),
//             reinterpret_cast<cutlass::half_t*>(c.mutable_data_ptr()), m, n,
//             k);
//     }
// }

template <const int kM, const int kN, const int kK>
using GemmShape = TileShape<kM, kN, kK>;

template <typename InType, typename AccType, typename WholeShape,
          typename CtaTileShape, const int kRK, typename WarpLayout>
struct KeGemmTraits {
    using BaseShape = traits::BaseTileShape<InType>;

    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;
    static constexpr int kWarpPerRow = tl::num_rows<WarpLayout>;
    static constexpr int kWarpPerCol = tl::num_cols<WarpLayout>;

    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;

    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;

    static const bool kSwizzled = true;

    // Total data access for operand A in global memory
    using GlobalA = GlobalTile<InType, tl::RowMajor<kTM, kK, kK>>;
    // Access a single global tile for operand A
    using GIteratorA = GTileIterator<GlobalA, TileShape<kTM, kTK>>;

    // Shared Tile for operand A
    using SharedA = SharedTile<InType, tl::RowMajor<kTM, kTK>, kSwizzled>;
    // Access a single register tile for operand A
    using SIteratorA = STileIterator<SharedA, TileShape<kTM, kRK>>;

    // Register tile for a single thread of operand A
    static constexpr int kAMs = kTM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kAKs = kRK / BaseShape::kTileSize;
    using RegA = RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAMs, kAKs>>;

    // Loaders for operand A
    using G2SLoaderA = GlobalToSharedLoader<SharedA, WarpLayout>;
    using S2RLoaderA =
        SharedToRegLoader<RegA, WarpLayout, WarpReuse::kRowReuseCont>;

    // Total data access for operand B in global memory
    using GlobalB = GlobalTile<InType, tl::ColMajor<kK, kTN, kK>>;
    // Access a single global tile for operand B
    using GIteratorB = GTileIterator<GlobalB, TileShape<kTK, kTN>>;

    // Shared Tile for operand B
    using SharedB = SharedTile<InType, tl::ColMajor<kTK, kTN>, kSwizzled>;
    // Access a single register tile for operand B
    using SIteratorB = STileIterator<SharedB, TileShape<kRK, kTN>>;

    static_assert(GIteratorA::sc1 == GIteratorB::sc0,
                  "mismatched K dimension!");
    static_assert(SIteratorA::sc1 == SIteratorB::sc0,
                  "mismatched K dimension!");

    // Register tile for a single thread of operand A
    static constexpr int kBKs = kRK / BaseShape::kTileSize;
    static constexpr int kBNs = kTN / kWarpPerCol / BaseShape::kTileSize;
    using RegB = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kBKs, kBNs>>;

    using G2SLoaderB = GlobalToSharedLoader<SharedB, WarpLayout>;
    using S2RLoaderB =
        SharedToRegLoader<RegB, WarpLayout, WarpReuse::kColReuseCont>;

    // Global Tile for output C
    using GlobalC = GlobalTile<AccType, tl::RowMajor<kTM, kTN, kN>>;
    // Shared Tile for output C
    using SharedC = SharedTile<AccType, tl::RowMajor<kTM, kTN>, kSwizzled>;

    // Register Tile for output C
    static constexpr int kCMs = kTM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kCNs = kTN / kWarpPerCol / BaseShape::kTileSize;
    using RegC = RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kCMs, kCNs>>;

    using R2SStorerC = RegToSharedStorer<RegC, WarpLayout>;
    using S2GStorerC = SharedToGlobalStorer<SharedC, WarpLayout>;
};

template <typename InType, typename AccType,                  //
          const int kM, const int kN, const int kK,           //
          const int kTM, const int kTN, const int kTK,        //
          typename GIteratorA, typename SIteratorA,           //
          typename SharedA, typename RegA,                    //
          typename G2SLoaderA, typename S2RLoaderA,           //
          typename GIteratorB, typename SIteratorB,           //
          typename SharedB, typename RegB,                    //
          typename G2SLoaderB, typename S2RLoaderB,           //
          typename GlobalC, typename SharedC, typename RegC,  //
          typename R2SStorerC, typename S2GStorerC>
__global__ void gemm(const InType* dA, const InType* dB, AccType* dC) {
    int offset_a = blockIdx.x * kTM * kK;
    int offset_b = blockIdx.y * kTN * kK;
    int offset_c = blockIdx.x * kTM * kN + blockIdx.y * kTN;

    extern __shared__ __align__(sizeof(double)) unsigned char buf[];
    InType* sA_ptr = reinterpret_cast<InType*>(buf);
    InType* sB_ptr = sA_ptr + SIteratorA::Tile::kNumel;
    AccType* sC_ptr = reinterpret_cast<AccType*>(buf);

    // declare tiles, iterators and loaders
    GIteratorA gAs(dA + offset_a);
    SIteratorA sAs(sA_ptr);

    GIteratorB gBs(dB + offset_b);
    SIteratorB sBs(sB_ptr);

    SharedA sA(sA_ptr);
    RegA rA;

    SharedB sB(sB_ptr);
    RegB rB;

    RegC acc;
    SharedC sC(sC_ptr);
    GlobalC gC(dC + offset_c);

    G2SLoaderA g2s_a;
    S2RLoaderA s2r_a;

    G2SLoaderB g2s_b;
    S2RLoaderB s2r_b;

    R2SStorerC r2s_c;
    S2GStorerC s2g_c;

    for (int k1 = 0; k1 < GIteratorA::sc1; ++k1) {
        g2s_a(gAs(k1), sA);
        g2s_b(gBs(k1), sB);
        __copy_async();
        __syncthreads();

        for (int k2 = 0; k2 < SIteratorA::sc1; ++k2) {
            s2r_a(sAs(k2), rA);
            s2r_b(sBs(k2), rB);

            compute::gemm_(rA, rB, acc);
        }
    }
    r2s_c(acc, sC);
    __syncthreads();
    s2g_c(sC, gC);
}

template <typename InType, typename AccType, typename WholeShape,
          typename CtaTileShape>
void run_gemm(const InType* dA, const InType* dB, AccType* dC) {
    using WarpLayout = tl::RowMajor<2, 2>;
    static constexpr int kRK = 32;

    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;

    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;

    using Config = KeGemmTraits<InType, AccType, WholeShape, CtaTileShape, kRK,
                                WarpLayout>;

    auto kernel =
        &gemm<InType, AccType, kM, kN, kK, kTM, kTN, kTK,
              typename Config::GIteratorA, typename Config::SIteratorA,
              typename Config::SharedA, typename Config::RegA,
              typename Config::G2SLoaderA, typename Config::S2RLoaderA,
              typename Config::GIteratorB, typename Config::SIteratorB,
              typename Config::SharedB, typename Config::RegB,
              typename Config::G2SLoaderB, typename Config::S2RLoaderB,
              typename Config::GlobalC, typename Config::SharedC,
              typename Config::RegC, typename Config::R2SStorerC,
              typename Config::S2GStorerC>;

    int block_x = CeilDiv<kM, kTM>;
    int block_y = CeilDiv<kN, kTN>;

    dim3 dim_grid(block_x, block_y, 1);
    dim3 dim_block(Config::kThreads, 1, 1);

    static constexpr int smem_size_inputs = kTK * (kTN + kTM) * sizeof(InType);
    static constexpr int smem_size_accumulators = kTM * kTN * sizeof(AccType);
    static constexpr int smem_size = smem_size_inputs > smem_size_accumulators
                                         ? smem_size_inputs
                                         : smem_size_accumulators;

    const int kMaxSmemPerBlock = 48 * 1024;
    if (smem_size > kMaxSmemPerBlock) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    kernel<<<dim_grid, dim_block, smem_size>>>(dA, dB, dC);
}

void custom_gemm_op(const torch::Tensor& A, const torch::Tensor& B,
                    torch::Tensor& C, int64_t m, int64_t n, int64_t k) {
    using InType = __half;
    using AccType = float;

    // TODO(Kuangjux): Fixed WholeShape and CtaTileShape.
    using WholeShape = GemmShape<4096, 4096, 4096>;
    using CtaTileShape = GemmShape<64, 256, 32>;

    auto dA = reinterpret_cast<const InType*>(A.data_ptr());
    auto dB = reinterpret_cast<const InType*>(B.data_ptr());
    auto dC = reinterpret_cast<AccType*>(C.data_ptr());

    run_gemm<InType, AccType, WholeShape, CtaTileShape>(dA, dB, dC);
}

}  // namespace tiledcuda::kernels
