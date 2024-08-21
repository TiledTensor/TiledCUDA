#pragma once

#include "cuda_utils.hpp"

__half max(__half a, __half b) { return a > b ? a : b; }

__half exp(__half x) { return __float2half(exp(__half2float(x))); }

void host_flash_attn(int kM, int kN, int kK, int kP, const __half* Q,
                     const __half* K, const __half* V, __half* O, __half* acc,
                     __half* exp_values, __half* cur_row_max,
                     __half* prev_row_max, __half* new_row_max,
                     __half* prev_norm_vec, __half* new_norm_vec,
                     __half* prev_sums, __half* cur_sums, __half* new_sums) {
    // Compute attention scores: Q * K^T
    for (int i = 0; i < kM; ++i) {
        for (int j = 0; j < kN; ++j) {
            __half s = 0.;
            for (int k = 0; k < kK; ++k) {
                s += Q[i * kK + k] * K[k + kK * j];
            }
            acc[i * kN + j] = s;
        }
    }

    // Compute row max of attention scores
    for (int i = 0; i < kM; ++i) {
        __half max_val = 0.;
        for (int j = 0; j < kN; ++j) {
            max_val = max(max_val, acc[i * kN + j]);
        }
        cur_row_max[i] = max_val;
    }

    // Broadcast sub row max to attention scores
    for (int i = 0; i < kM; ++i) {
        for (int j = 0; j < kN; ++j) {
            acc[i * kN + j] = exp(acc[i * kN + j] - cur_row_max[i]);
        }
    }

    // Compute reduce sum for each row and store into `cur_sums`.
    for (int i = 0; i < kM; ++i) {
        __half sum = 0.;
        for (int j = 0; j < kN; ++j) {
            sum += acc[i * kN + j];
        }
        cur_sums[i] = sum;
    }

    // Compare cur row max with prev row max and compute new row max
    for (int i = 0; i < kM; ++i) {
        new_row_max[i] = max(cur_row_max[i], prev_row_max[i]);
    }

    for (int i = 0; i < kM; ++i) {
        // Compute remormalization factor for the previous block.
        prev_norm_vec[i] = exp(prev_row_max[i] - new_row_max[i]);

        // Compute remormalization factor for the current block.
        new_norm_vec[i] = exp(cur_row_max[i] - new_row_max[i]);
    }

    // Update normalization factor l(x).
    for (int i = 0; i < kM; ++i) {
        new_sums[i] =
            prev_norm_vec[i] * prev_sums[i] + new_norm_vec[i] * cur_sums[i];
    }

    // Compute unnormalizatied attention score @ values
    for (int i = 0; i < kM; ++i) {
        for (int j = 0; j < kP; ++j) {
            __half s = 0.;
            for (int k = 0; k < kN; ++k) {
                s += acc[i * kN + k] * V[k + kN * j];
            }
            exp_values[i * kP + j] = s;
        }
    }

    // Compute O = (O * prev_sums * prev_norm_vec + new_norm_vec * exp_values) /
    // new_sums.
    for (int i = 0; i < kM; ++i) {
        for (int j = 0; j < kP; ++j) {
            O[i * kP + j] = (O[i * kP + j] * prev_sums[i] * prev_norm_vec[i] +
                             new_norm_vec[i] * exp_values[i * kP + j]) /
                            new_sums[i];
        }
    }
}