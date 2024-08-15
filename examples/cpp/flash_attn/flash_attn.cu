#include "util.hpp"

template <typename InType, typename OutType, typename GlobalQ, typename RegQ,
          typename LoaderQ, typename IteratorK, typename RegK, typename LoaderK,
          typename IteratorV, typename RegV, typename LoaderV, typename RegAcc,
          typename RegAccHalf, typename RegO, typename RegVec, typename Cast,
          typename CopyVec, typename ReduceMax, typename BroadcastSub,
          typename BroadcastMul, typename BroadcastDiv, typename BlockAdd,
          typename BlockExp, typename VecMax, typename VecAdd, typename VecSub,
          typename VecExp, typename VecMul>
__global__ void flash_attn(const InType* dQ, const InType* dK, const InType* dV,
                           OutType* dO) {
    GlobalQ gQ(dQ);
    RegQ rQ;
    LoaderQ loader_q;

    IteratorK iter_g2r_k(dK);
    RegK rK;
    LoaderK loader_k;

    IteratorV iter_g2r_v(dV);
    RegV rV;
    LoaderV loader_v;

    RegAcc attn_block_f32;
    RegAccHalf attn_block;

    RegO unnormized_attn_block;
    RegO rO;

    RegVec prev_reg_norm;
    RegVec cur_reg_norm;

    RegVec prev_reg_renorm;
    RegVec cur_reg_renorm;

    RegVec prev_reg_max;
    RegVec cur_reg_max;
    RegVec new_reg_max;

    RegVec prev_reg_sub;
    RegVec cur_reg_sub;

    RegVec update_prev_reg_norm;
    RegVec update_cur_reg_norm;

    RegVec prev_sum_reg;
    RegVec new_sum_reg;

    RegVec prev_o_reg;
    RegVec cur_o_reg;

    ReduceMax row_max;
    Cast cast;
    CopyVec copy_vec;

    BroadcastSub broadcast_sub;
    BroadcastMul broadcast_mul;
    BroadcastDiv broadcast_div;

    BlockExp block_exp;
    BlockAdd block_add;

    VecMax vec_max;
    VecAdd vec_add;
    VecSub vec_sub;
    VecExp vec_exp;
    VecMul vec_mul;

    // Load Q tiles from global to register.
    loader_q(gQ, rQ);

    for (int tc = 0; tc < IteratorK::sc1; ++tc) {
        // Iterate over tc.
        // load K, V tiles from global to register.
        // rK: [d, B_c]
        loader_k(iter_g2r_k(k), rK);
        // rV: [B_c, d]
        loader_v(iter_g2r_v(k), rV);
        __syncthreads();

        // Q @ K.T
        // reg_acc_0: [B_r, d]
        compute::gemm_(rQ, rK, attn_block_f32);
        __syncthreads();

        // Cast float into half.
        cast(attn_block_f32, attn_block);

        // Copy `reg_max`, `reg_norm` into `last_reg_max`, `last_reg_norm`.
        copy_vec(cur_reg_max, prev_reg_max);
        copy_vec(cur_reg_norm, prev_reg_norm);

        // Compute row max.
        row_max(attn_block, reg_max);
        // Broadcast subtract from `attn_block`.
        // (lhs, rhs, dst)
        broadcast_sub(attn_block, cur_reg_max, attn_block);

        // Compute exp in `attn_block`.
        block_exp(attn_block);

        // Compute new max vector.
        vec_max(cur_reg_max, prev_reg_max, new_reg_max);

        // Renormalization for the previous block.
        vec_sub(prev_reg_max, new_reg_max, prev_reg_sub);
        vec_exp(prev_reg_sub, prev_reg_renorm);

        // Renormalization for the current block.
        vec_sub(cur_reg_max, new_reg_max, cur_reg_sub);
        vec_exp(cur_reg_sub, cur_reg_renorm);

        // Update normalization.
        copy_vec(prev_sum_reg, new_sum_reg);
        vec_mul(prev_reg_norm, prev_reg_renorm, update_prev_reg_norm);
        vec_mul(cur_reg_norm, cur_reg_renorm, update_cur_reg_norm);
        vec_add(update_prev_reg_norm, update_cur_reg_norm, new_sum_reg);

        // Compute O.
        // Compute Unnormized Attention Block.
        compute::gemm_(attn_block, rV, unnormized_attn_block);
        vec_mul(prev_sum_reg, prev_reg_norm, prev_o_reg);
        broadcast_mul(rO, prev_o_reg, rO);
        broadcast_mul(unnormized_attn_block, cur_reg_norm,
                      unnormized_attn_block);
        block_add(rO, unnormized_attn_block, rO);
        broadcast_mul(rO, new_sum_reg, rO);
    }
}

int main() {}
