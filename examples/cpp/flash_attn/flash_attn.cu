#include "flash_attn.hpp"
#include "util.hpp"

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
    host_flash_attn(kM, kN, kK, kP, kBatch,
                    thrust::raw_pointer_cast(h_a.data()),
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
        FlashAttentionShape<64 /*kTM*/, 64 /*kTN*/, 128 /*kTK*/, 128
                            /*kTP*/>,
        1>();

    run<FlashAttentionShape<64 /*M*/, 64 /*N*/, 128 /*K*/, 128 /*P*/>,
        FlashAttentionShape<64 /*kTM*/, 64 /*kTN*/, 128 /*kTK*/, 128
                            /*kTP*/>,
        2>();

    run<FlashAttentionShape<64 /*M*/, 128 /*N*/, 128 /*K*/, 128 /*P*/>,
        FlashAttentionShape<64 /*kTM*/, 64 /*kTN*/, 128 /*kTK*/, 128
                            /*kTP*/>,
        1>();

    run<FlashAttentionShape<64 /*M*/, 256 /*N*/, 128 /*K*/, 128 /*P*/>,
        FlashAttentionShape<64 /*kTM*/, 64 /*kTN*/, 128 /*kTK*/, 128 /*kTP*/>,
        1>();

    run<FlashAttentionShape<64 /*M*/, 512 /*N*/, 128 /*K*/, 128 /*P*/>,
        FlashAttentionShape<64 /*kTM*/, 64 /*kTN*/, 128 /*kTK*/, 128 /*kTP*/>,
        1>();

    return 0;
}
