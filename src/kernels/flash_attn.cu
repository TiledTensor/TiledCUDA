#include "cell/compute/mod.hpp"
#include "cell/mod.hpp"
#include "kernels/flash_attn.hpp"
#include "types/mod.hpp"
#include "util/debug.hpp"
#include "util/print.hpp"

using namespace tiledcuda;
using namespace tiledcuda::cell;
using namespace tiledcuda::cell::copy;
namespace tl = tile_layout;

namespace tiledcuda::kernels {

template <const int kM, const int kN, const int kK, const int kP>
using FlashAttentionShape = TileShape<kM, kN, kK, kP>;

template <typename InType, typename AccType, typename OutType,
          typename WholeShape, typename CtaTileShape>
struct FlashAttentionTraits {
    using BaseShape = traits::BaseTileShape<InType>;

    using WarpLayout = tl::RowMajor<4, 1>;

    static constexpr int kWarpPerRow = tl::num_rows<WarpLayout>;
    static constexpr int kWarpPerCol = tl::num_cols<WarpLayout>;
    static_assert(kWarpPerCol == 1, "WarpPerCol must be 1");

    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;

    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;
    static constexpr int kP = dim_size<3, WholeShape>;

    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;
    static constexpr int kTP = dim_size<3, CtaTileShape>;

    // operand A
    using GlobalA = GlobalTile<InType, tl::RowMajor<kTM, kK>>;
    // chunk the K dimension to fit into shared memory
    using GIteratorA = GTileIterator<GlobalA, TileShape<kTM, kTK>>;

    using SharedA = SharedTile<InType, tl::RowMajor<kTM, kTK>, true>;

    static constexpr int kAMs = kTM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kAKs = kTK / BaseShape::kTileSize;
    using RegA = RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAMs, kAKs>>;

    using SharedALoader = GlobalToSharedLoader<SharedA, WarpLayout>;
    using RegALoader =
        SharedToRegLoader<RegA, WarpLayout, WarpReuse::kRowReuseCont>;

    // operand B
    using GlobalB = GlobalTile<InType, tl::ColMajor<kK, kN>>;
    using GIteratorB = GTileIterator<GlobalB, TileShape<kTK, kTN>>;
    using SharedB = SharedTile<InType, tl::ColMajor<kTK, kTN>, true>;

    static constexpr int kBKs = kTK / BaseShape::kTileSize;
    static constexpr int kBNs = kTN / kWarpPerCol / BaseShape::kTileSize;
    using RegB = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kBKs, kBNs>>;

    using SharedBLoader = GlobalToSharedLoader<SharedB, WarpLayout>;
    using RegBLoader =
        SharedToRegLoader<RegB, WarpLayout, WarpReuse::kColReuseCont>;

    // operand C
    using GlobalC = GlobalTile<InType, tl::ColMajor<kN, kTP>>;
    // chunk the N dimension to fit into shared memory
    using GIteratorC = GTileIterator<GlobalC, TileShape<kTN, kTP>>;
    using SharedC = SharedTile<InType, tl::ColMajor<kTN, kTP>, true>;

    static constexpr int kCNs = kTN / BaseShape::kTileSize;
    static constexpr int kCPs = kTP / kWarpPerCol / BaseShape::kTileSize;
    using RegC = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kCNs, kCPs>>;

    using SharedCLoader = GlobalToSharedLoader<SharedC, WarpLayout>;
    using RegCLoader =
        SharedToRegLoader<RegC, WarpLayout, WarpReuse::kColReuseCont>;

    // output D
    using GlobalD = GlobalTile<OutType, tl::RowMajor<kTM, kTP>>;

    static constexpr int kDMs = kTM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kDPs = kTP / kWarpPerCol / BaseShape::kTileSize;
    using RegD = RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kDMs, kDPs>>;
    using RegDCast =
        RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kDMs, kDPs>>;
    using DStorer = copy::RegToGlobalStorer<GlobalD, RegDCast, WarpLayout>;

    static constexpr int kAccMs = kTM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kAccNs = kTN / kWarpPerCol / BaseShape::kTileSize;

    // Reg Acc
    using RegAcc =
        RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kAccMs, kAccNs>>;
    using RegAccCast =
        RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAccMs, kAccNs>>;

    using RegAccPrinter =
        cell::RegTilePrinter<RegAccCast, tl::Layout::kRowMajor>;

    // Convert the accumulator to half
    using ConvertHalf = compute::RegTileConvert<RegAcc, RegAccCast>;
    using ConvertO = compute::RegTileConvert<RegD, RegDCast>;

    using RegVec = RegTile<InType, tl::RowMajor<kAccMs, 2>>;
    using RegVecPrinter = cell::RegVecPrinter<RegVec>;

    using CopyVec = copy::BaseTileCopy<RegVec>;
    using RowMax = compute::MaxReduce<RegAccCast, tl::Layout::kRowMajor>;

    using RowSum = compute::SumReduce<RegAccCast, tl::Layout::kRowMajor>;

    using BroadcastSub =
        compute::BroadcastSub<RegVec, RegAccCast, tl::Layout::kRowMajor>;
    using BroadcastMul =
        compute::BroadcastMul<RegVec, RegDCast, tl::Layout::kRowMajor>;
    using BroadcastDiv =
        compute::BroadcastDiv<RegVec, RegDCast, tl::Layout::kRowMajor>;

    using BlockExp = compute::RegTileExp<RegAccCast>;
    using BlockAdd = compute::RegTileAdd<RegDCast>;

    using VecMax = compute::BaseTileMax<RegVec>;
    using VecAdd = compute::BaseTileAdd<RegVec>;
    using VecSub = compute::BaseTileSub<RegVec>;
    using VecMul = compute::BaseTileMul<RegVec>;
    using VecExp = compute::BaseTileExp<RegVec>;
};

template <typename InType,
          typename AccType,                                      //
          typename OutType,                                      //
          typename GIteratorQ, typename SharedQ, typename RegQ,  //
          typename SharedQLoader, typename RegQLoader,           //
          typename GIteratorK, typename SharedK, typename RegK,  //
          typename SharedKLoader, typename RegKLoader,           //
          typename GIteratorV, typename SharedV, typename RegV,  //
          typename SharedVLoader, typename RegVLoader,           //
          typename RegAcc, typename RegAccCast, typename GlobalO, typename RegO,
          typename RegOCast, typename OStorer, typename ConvertAcc,
          typename ConvertO, typename RegVec, typename CopyVec, typename RowMax,
          typename RowSum, typename BroadcastSub, typename BroadcastMul,
          typename BroadcastDiv, typename BlockExp, typename BlockAdd,
          typename VecMax, typename VecAdd, typename VecSub, typename VecMul,
          typename VecExp, typename RegVecPrinter, typename RegAccPrinter>
__global__ void flash_attention(const InType* dQ, const InType* dK,
                                const InType* dV, InType* dO, int kM, int kN,
                                int kK, int kP, int kTM, int kTN, int kTK,
                                int kTP) {
    // Advance to the global data tile to the current CTA.
    const InType* Q = dQ + blockIdx.z * (kM * kK) + blockIdx.x * (kTM * kK);
    const InType* K = dK + blockIdx.z * (kK * kN);
    const InType* gV_ptr =
        dV + blockIdx.z * (kN * kP) + blockIdx.y * (kTP * kN);

    OutType* gO_ptr = dO + blockIdx.z * (kM * kP) + blockIdx.x * (kTM * kP) +
                      (blockIdx.y * kTP);

    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<InType*>(shared_buf);

    InType* sQ_ptr = shm;
    InType* sK_ptr = shm + SharedQ::kNumel;
    InType* sV_ptr = shm + SharedQ::kNumel + SharedK::kNumel;

    GIteratorQ gQs(Q);
    SharedQ sQ(sQ_ptr);
    RegQ rQ;

    SharedQLoader load_sq;
    RegQLoader load_rq;

    GIteratorK gKs(K);
    SharedK sK(sK_ptr);
    RegK rK;

    SharedKLoader load_sk;
    RegKLoader load_rk;

    GIteratorV gVs(gV_ptr);
    SharedV sV(sV_ptr);

    SharedVLoader load_sv;
    RegVLoader load_rv;
    RegV rV;

    RegO exp_values_f32;

    RegOCast rO;
    RegOCast exp_values;

    RegAcc attn_block_f32;
    RegAccCast attn_block;

    RegVecPrinter print_vec;
    RegAccPrinter print_acc;

    RegVec prev_norm_vec;
    RegVec cur_norm_vec;

    RegVec prev_max_vec;
    RegVec cur_max_vec;
    RegVec new_max_vec;

    RegVec prev_sum_vec;
    RegVec cur_sum_vec;
    RegVec new_sum_vec;

    RegVec prev_norm_mul_sum;
    RegVec cur_norm_mul_sum;
    RegVec prev_sum_mul_norm;

    RowMax row_max;
    RowSum row_sum;
    CopyVec copy_vec;

    ConvertAcc cast_acc;  // Convert acc to half precision
    ConvertO cast_o;      // Convert half precision to float.

    BroadcastSub broadcast_sub;
    BroadcastMul broadcast_mul;
    BroadcastDiv broadcast_div;

    BlockExp block_exp;
    BlockAdd block_add;

    VecMax vec_max;
    VecAdd vec_add;
    VecSub vec_sub;
    VecMul vec_mul;
    VecExp vec_exp;

    for (int n = 0; n < GIteratorV::sc0; ++n) {
        load_sv(gVs(n), sV);

        for (int k = 0; k < GIteratorQ::sc1; ++k) {
            load_sq(gQs(k), sQ);
            load_sk(gKs(k, n), sK);
            __copy_async();
            __syncthreads();

            load_rq(sQ, rQ);
            load_rk(sK, rK);
            __syncthreads();

            compute::gemm(rQ, rK, attn_block_f32);
        }
        load_rv(sV, rV);
        __syncthreads();

        cast_acc(attn_block_f32, attn_block);

        // Compute row max.
        row_max(attn_block, cur_max_vec);

        // Broadcast subtract from `attn_block`.
        broadcast_sub(cur_max_vec, attn_block);

        // Compute exp in `attn_block`.
        block_exp(attn_block, attn_block);

        // Compute `cur_sum_vec` by reduce sum of `attn_block`.
        row_sum(attn_block, cur_sum_vec);

        // Compute new max vector.
        vec_max(cur_max_vec, prev_max_vec, new_max_vec);

        // Renormalization for the previous block.
        vec_sub(prev_max_vec, new_max_vec, prev_norm_vec);
        vec_exp(prev_norm_vec, prev_norm_vec);

        // Renormalization for the current block.
        vec_sub(cur_max_vec, new_max_vec, cur_norm_vec);
        vec_exp(cur_norm_vec, cur_norm_vec);

        // Update normalization factor l(x)
        vec_mul(prev_norm_vec, prev_sum_vec, prev_norm_mul_sum);
        vec_mul(cur_norm_vec, cur_sum_vec, cur_norm_mul_sum);
        vec_add(prev_norm_mul_sum, cur_norm_mul_sum, new_sum_vec);

        // Compute unnormized attention block.
        compute::gemm(attn_block, rV, exp_values_f32);

        cast_o(exp_values_f32, exp_values);

        broadcast_mul(prev_norm_mul_sum, rO);

        broadcast_mul(cur_norm_vec, exp_values);

        block_add(rO, exp_values, rO);

        // Normalize the attention block.
        broadcast_div(new_sum_vec, rO);

        // Update max vector and sum vector.
        copy_vec(new_max_vec, prev_max_vec);
        copy_vec(new_sum_vec, prev_sum_vec);

        // Clear the accumulator.
        attn_block_f32.clear();
        exp_values_f32.clear();
    }
    __syncthreads();

    GlobalO gO(gO_ptr);
    OStorer storer_o;  // Store O tile from register to global.
    storer_o(rO, gO);
}

template <typename InType, typename AccType, typename OutType,
          typename WholeShape, typename CtaTileShape, const int kBatch>
void run_flash_attention(const InType* dQ, const InType* dK, const InType* dV,
                         OutType* dO) {
    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;
    static constexpr int kP = dim_size<3, WholeShape>;

    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;
    static constexpr int kTP = dim_size<3, CtaTileShape>;

    static_assert(kK == kTK,
                  "The current implementation requires kTK == K for now.");
    static_assert(kP == kTP,
                  "The current implementation requires kTP == P for now.");

    using Config = FlashAttentionTraits<InType, AccType, OutType, WholeShape,
                                        CtaTileShape>;

    using RegA = typename Config::RegA;
    using RegB = typename Config::RegB;
    using RegC = typename Config::RegC;
    using RegD = typename Config::RegD;
    using RegDCast = typename Config::RegDCast;
    using RegAcc = typename Config::RegAcc;
    using RegAccCast = typename Config::RegAccCast;

    using RegAccPrinter = typename Config::RegAccPrinter;

    using GIteratorA = typename Config::GIteratorA;
    using SharedA = typename Config::SharedA;
    using SharedALoader = typename Config::SharedALoader;
    using RegALoader = typename Config::RegALoader;

    using GIteratorB = typename Config::GIteratorB;
    using SharedB = typename Config::SharedB;
    using SharedBLoader = typename Config::SharedBLoader;
    using RegBLoader = typename Config::RegBLoader;

    using GIteratorC = typename Config::GIteratorC;
    using SharedC = typename Config::SharedC;
    using SharedCLoader = typename Config::SharedCLoader;
    using RegCLoader = typename Config::RegCLoader;

    using DStorer = typename Config::DStorer;

    using ConvertAcc = typename Config::ConvertHalf;
    using ConvertO = typename Config::ConvertO;

    using RegVec = typename Config::RegVec;
    using RegVecPrinter = typename Config::RegVecPrinter;

    using CopyVec = typename Config::CopyVec;
    using RowMax = typename Config::RowMax;
    using RowSum = typename Config::RowSum;

    using BroadcastSub = typename Config::BroadcastSub;
    using BroadcastMul = typename Config::BroadcastMul;
    using BroadcastDiv = typename Config::BroadcastDiv;

    using BlockExp = typename Config::BlockExp;
    using BlockAdd = typename Config::BlockAdd;

    using VecMax = typename Config::VecMax;
    using VecAdd = typename Config::VecAdd;
    using VecSub = typename Config::VecSub;
    using VecMul = typename Config::VecMul;
    using VecExp = typename Config::VecExp;

    int block_x = CeilDiv<kM, kTM>;
    int block_y = CeilDiv<kP, kTP>;
    int block_z = kBatch;

    dim3 grid(block_x, block_y, block_z);
    dim3 block(Config::kThreads, 1, 1);

    int shm_input = (kTM * kTK + kTK * kTN + kTN * kTP);
    int shm_output = kTM * kTP;
    int shm_size = shm_input < shm_output ? shm_output * sizeof(InType)
                                          : shm_input * sizeof(InType);

    auto kernel =
        &flash_attention<InType, AccType,
                         OutType,                    //
                         GIteratorA, SharedA, RegA,  //
                         SharedALoader, RegALoader,  //
                         GIteratorB, SharedB, RegB,  //
                         SharedBLoader, RegBLoader,  //
                         GIteratorC, SharedC, RegC,  //
                         SharedCLoader, RegCLoader,  //
                         RegAcc, RegAccCast, typename Config::GlobalD, RegD,
                         RegDCast, DStorer, ConvertAcc, ConvertO, RegVec,
                         CopyVec, RowMax, RowSum, BroadcastSub, BroadcastMul,
                         BroadcastDiv, BlockExp, BlockAdd, VecMax, VecAdd,
                         VecSub, VecMul, VecExp, RegVecPrinter, RegAccPrinter>;

    if (shm_size > 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    }

    kernel<<<grid, block, shm_size, 0>>>(dQ, dK, dV, dO, kM, kN, kK, kP, kTM,
                                         kTN, kTK, kTP);

    cudaDeviceSynchronize();
}

void custom_flash_attention_op(const torch::Tensor& Q, const torch::Tensor& K,
                               const torch::Tensor& V, torch::Tensor& O,
                               int64_t m, int64_t n, int64_t k, int64_t p) {
    using InType = __half;
    using AccType = float;
    using OutType = __half;

    using CtaTileShape = TileShape<64, 64, 128, 128>;

    auto dQ = reinterpret_cast<const InType*>(Q.data_ptr());
    auto dK = reinterpret_cast<const InType*>(K.data_ptr());
    auto dV = reinterpret_cast<const InType*>(V.data_ptr());
    auto dO = reinterpret_cast<OutType*>(O.data_ptr());

    if (m == 64 && n == 256 && k == 128 && p == 128) {
        using WholeShape = FlashAttentionShape<64, 256, 128, 128>;
        const int kBatch = 1;
        run_flash_attention<InType, AccType, OutType, WholeShape, CtaTileShape,
                            kBatch>(dQ, dK, dV, dO);
    } else if (m == 64 && n == 128 && k == 128 && p == 128) {
        using WholeShape = FlashAttentionShape<64, 128, 128, 128>;
        const int kBatch = 1;
        run_flash_attention<InType, AccType, OutType, WholeShape, CtaTileShape,
                            kBatch>(dQ, dK, dV, dO);
    } else if (m == 64 && n == 64 && k == 128 && p == 128) {
        using WholeShape = FlashAttentionShape<64, 64, 128, 128>;
        const int kBatch = 1;
        run_flash_attention<InType, AccType, OutType, WholeShape, CtaTileShape,
                            kBatch>(dQ, dK, dV, dO);
    } else {
        throw std::runtime_error("Unsupported shape");
    }
}
}  // namespace tiledcuda::kernels
