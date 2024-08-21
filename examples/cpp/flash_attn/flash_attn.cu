#include "util.hpp"
#include "util/cuda_timer.hpp"

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
          typename VecExp>
__global__ void KeFlashAttention(const InType* dQ, const InType* dK,
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

    RegO unnormized_attn_block_f32;

    RegOCast rO;
    RegOCast unnormized_attn_block;

    RegAcc attn_block_f32;
    RegAccCast attn_block;

    RegVec prev_norm_vec;
    RegVec cur_norm_vec;

    RegVec prev_max_vec;
    RegVec cur_max_vec;
    RegVec new_max_vec;

    RegVec prev_sum_vec;
    RegVec cur_sum_vec;
    RegVec new_sum_vec;

    RowMax row_max;
    RowSum row_sum;
    CopyVec copy_vec;

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

            compute::gemm_(rQ, rK, attn_block_f32);
        }
        load_rv(sV, rV);
        __syncthreads();

        ConvertAcc cast_acc;  // Convert acc to half precision
        cast_acc(attn_block_f32, attn_block);

#ifdef DEBUG
        if (tiledcuda::thread(0)) {
            printf("Thread 0 attn block: \n");
            for (int h = 0; h < RegAccCast::kRows; ++h) {
                for (int w = 0; w < RegAccCast::kCols; ++w) {
                    printf("(%d, %d):\n", h, w);
                    attn_block(h, w).dump_value();
                }
            }
        }
#endif

        // Copy `cur_max_vec`, `cur_norm_vec` into `prev_max_vec`,
        // `prev_norm_vec`.
        copy_vec(cur_max_vec, prev_max_vec);
        copy_vec(cur_norm_vec, prev_norm_vec);

        // Compute row max.
        row_max(attn_block, cur_max_vec);

#ifdef DEBUG
        if (tiledcuda::thread(0)) {
            // WarpLayout: (4, 1)
            printf("Thread 0 cur_max_vec: \n");
            cur_max_vec.dump_value();
        }
#endif

        // Broadcast subtract from `attn_block`.
        broadcast_sub(cur_max_vec, attn_block);

#ifdef DEBUG
        if (tiledcuda::thread(0)) {
            printf("Thread 0 attn block after broadcast sub: \n");
            for (int h = 0; h < RegAccCast::kRows; ++h) {
                for (int w = 0; w < RegAccCast::kCols; ++w) {
                    printf("(%d, %d):\n", h, w);
                    attn_block(h, w).dump_value();
                }
            }
        }
#endif

        // Compute exp in `attn_block`.
        block_exp(attn_block, attn_block);

        // Compute `cur_sum_vec` by reduce sum of `attn_block`.
        row_sum(attn_block, cur_sum_vec);

        // Compute new max vector.
        vec_max(cur_max_vec, prev_max_vec, new_max_vec);

#ifdef DEBUG
        if (tiledcuda::thread(0)) {
            printf("Thread 0 new_max_vec: \n");
            new_max_vec.dump_value();
        }

        if (tiledcuda::thread(4)) {
            printf("Thread 4 new_max_vec: \n");
            new_max_vec.dump_value();
        }
#endif

        // Renormalization for the previous block.
        vec_sub(prev_max_vec, new_max_vec, prev_norm_vec);
        vec_exp(prev_norm_vec, prev_norm_vec);

        // Renormalization for the current block.
        vec_sub(cur_max_vec, new_max_vec, cur_norm_vec);
        vec_exp(cur_norm_vec, cur_norm_vec);

#ifdef DEBUG
        if (tiledcuda::thread(0)) {
            printf("Thread 0 prev_max_vec: \n");
            prev_max_vec.dump_value();
            printf("Thread 0 cur_max_vec: \n");
            cur_max_vec.dump_value();
            printf("Thread 0 prev_norm_vec: \n");
            prev_norm_vec.dump_value();
            printf("Thread 0 cur_norm_vec: \n");
            cur_norm_vec.dump_value();
        }
#endif

        // Update normalization factor l(x)
        vec_mul(prev_norm_vec, prev_sum_vec, prev_sum_vec);
        vec_mul(cur_norm_vec, cur_sum_vec, cur_sum_vec);
        vec_add(prev_sum_vec, cur_sum_vec, new_sum_vec);

#ifdef DEBUG
        if (tiledcuda::thread(0)) {
            printf("Thread 0 prev_sum_vec: \n");
            prev_sum_vec.dump_value();
            printf("Thread 0 cur_sum_vec: \n");
            cur_sum_vec.dump_value();
            printf("Thread 0 new_sum_vec: \n");
            new_sum_vec.dump_value();
        }
#endif

        // Compute unnormized attention block.
        compute::gemm_(attn_block, rV, unnormized_attn_block_f32);

        __syncthreads();

        ConvertO cast_o;  // Convert half precision to float.
        cast_o(unnormized_attn_block_f32, unnormized_attn_block);

#ifdef DEBUG
        if (tiledcuda::thread(0)) {
            printf("Thread 0 unnormized_attn_block: \n");
            for (int h = 0; h < RegOCast::kRows; ++h) {
                for (int w = 0; w < RegOCast::kCols; ++w) {
                    printf("(%d, %d):\n", h, w);
                    unnormized_attn_block(h, w).dump_value();
                }
            }
        }
#endif

        vec_mul(prev_sum_vec, prev_norm_vec, prev_norm_vec);
        broadcast_mul(prev_norm_vec, rO);

        broadcast_mul(cur_norm_vec, unnormized_attn_block);

        block_add(rO, unnormized_attn_block, rO);

        broadcast_div(new_sum_vec, rO);

        // Cear the accumulator.
        attn_block_f32.clear();

        // Update max vector and sum vector.
        copy_vec(new_max_vec, prev_max_vec);
        copy_vec(new_sum_vec, prev_sum_vec);
    }
    __syncthreads();

    GlobalO gO(gO_ptr);
    OStorer storer_o;  // Store O tile from register to global.
    storer_o(rO, gO);
}

template <typename WholeShape, typename CtaTileShape, const int kBatch>
void run(bool check = true) {
    using InType = __half;
    using AccType = float;
    using OutType = __half;

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

    // initialize data
    thrust::host_vector<InType> h_a(kM * kK * kBatch);

    for (int i = 0; i < h_a.size(); ++i)
        h_a[i] = static_cast<InType>(rand_float());

    thrust::host_vector<InType> h_b(kK * kN * kBatch);
    for (int i = 0; i < h_b.size(); ++i)
        h_b[i] = static_cast<InType>(rand_float());

    thrust::host_vector<InType> h_c(kN * kP * kBatch);
    for (int i = 0; i < h_c.size(); ++i)
        h_c[i] = static_cast<InType>(rand_float());

    thrust::host_vector<InType> h_d(kM * kP * kBatch);
    thrust::fill(h_d.begin(), h_d.end(), 0.);

    // Host side memory initialization.
    thrust::host_vector<InType> acc(kM * kN * kBatch);
    thrust::fill(acc.begin(), acc.end(), 0.);

    thrust::host_vector<InType> exp_values(kM * kP * kBatch);
    thrust::fill(exp_values.begin(), exp_values.end(), 0.);

    thrust::host_vector<InType> h_o(kM * kP * kBatch);
    thrust::fill(h_o.begin(), h_o.end(), 0.);

    thrust::host_vector<InType> cur_row_max(kM * kBatch);
    thrust::fill(cur_row_max.begin(), cur_row_max.end(), 0.);

    thrust::host_vector<InType> prev_row_max(kM * kBatch);
    thrust::fill(prev_row_max.begin(), prev_row_max.end(), 0.);

    thrust::host_vector<InType> new_row_max(kM * kBatch);
    thrust::fill(new_row_max.begin(), new_row_max.end(), 0.);

    thrust::host_vector<InType> prev_norm_vec(kM * kBatch);
    thrust::fill(prev_norm_vec.begin(), prev_norm_vec.end(), 0.);

    thrust::host_vector<InType> new_norm_vec(kM * kBatch);
    thrust::fill(new_norm_vec.begin(), new_norm_vec.end(), 0.);

    thrust::host_vector<InType> prev_sum_vec(kM * kBatch);
    thrust::fill(prev_sum_vec.begin(), prev_sum_vec.end(), 0.);

    thrust::host_vector<InType> cur_sum_vec(kM * kBatch);
    thrust::fill(cur_sum_vec.begin(), cur_sum_vec.end(), 0.);

    thrust::host_vector<InType> new_sum_vec(kM * kBatch);
    thrust::fill(new_sum_vec.begin(), new_sum_vec.end(), 0.);

    thrust::device_vector<InType> d_a = h_a;
    thrust::device_vector<InType> d_b = h_b;
    thrust::device_vector<InType> d_c = h_c;
    thrust::device_vector<InType> d_d = h_d;

    const InType* A = thrust::raw_pointer_cast(d_a.data());
    const InType* B = thrust::raw_pointer_cast(d_b.data());
    const InType* C = thrust::raw_pointer_cast(d_c.data());
    InType* D = thrust::raw_pointer_cast(d_d.data());

    using Config =
        FlashAttentionTraits<InType, AccType, WholeShape, CtaTileShape>;

    using RegA = typename Config::RegA;
    using RegB = typename Config::RegB;
    using RegC = typename Config::RegC;
    using RegD = typename Config::RegD;
    using RegDCast = typename Config::RegDCast;
    using RegAcc = typename Config::RegAcc;
    using RegAccCast = typename Config::RegAccCast;

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
        &KeFlashAttention<InType, AccType,
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
                          VecSub, VecMul, VecExp>;

    if (shm_size > 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    }

    kernel<<<grid, block, shm_size, 0>>>(A, B, C, D, kM, kN, kK, kP, kTM, kTN,
                                         kTK, kTP);

    cudaDeviceSynchronize();

    // Call host-side reference implementation.
    host_flash_attn(kM, kN, kK, kP, thrust::raw_pointer_cast(h_a.data()),
                    thrust::raw_pointer_cast(h_b.data()),
                    thrust::raw_pointer_cast(h_c.data()),
                    thrust::raw_pointer_cast(h_o.data()),
                    thrust::raw_pointer_cast(acc.data()),
                    thrust::raw_pointer_cast(exp_values.data()),
                    thrust::raw_pointer_cast(cur_row_max.data()),
                    thrust::raw_pointer_cast(prev_row_max.data()),
                    thrust::raw_pointer_cast(new_row_max.data()),
                    thrust::raw_pointer_cast(prev_norm_vec.data()),
                    thrust::raw_pointer_cast(new_norm_vec.data()),
                    thrust::raw_pointer_cast(prev_sum_vec.data()),
                    thrust::raw_pointer_cast(cur_sum_vec.data()),
                    thrust::raw_pointer_cast(new_sum_vec.data()));

    h_d = d_d;

    if (check_results(thrust::raw_pointer_cast(h_o.data()),
                      thrust::raw_pointer_cast(h_d.data()), kM * kP * kBatch)) {
        std::cout << "Test passed." << std::endl;
    } else {
        std::cout << "Test failed." << std::endl;
    }
}

int main() {
    run<FlashAttentionShape<64 /*M*/, 64 /*N*/, 128 /*K*/, 128 /*P*/>,
        FlashAttentionShape<64 /*kTM*/, 64 /*kTN*/, 128 /*kTK*/, 128 /*kTP*/>,
        1>();

    return 0;
}
