#pragma once

#include "cuda_utils.hpp"

__half max(__half a, __half b) { return a > b ? a : b; }

__half exp(__half x) { return __float2half(exp(__half2float(x))); }

void host_flash_attn(int kM, int kN, int kK, int kP, int kBatch,
                     const __half* Q, const __half* K, const __half* V,
                     __half* O, __half* acc, __half* exp_values,
                     __half* cur_row_max, __half* prev_row_max,
                     __half* new_row_max, __half* prev_norm_vec,
                     __half* new_norm_vec, __half* prev_sums, __half* cur_sums,
                     __half* new_sums) {
#pragma omp parallel for
    for (int b = 0; b < kBatch; ++b) {
        const __half* q_ = Q + b * kM * kK;
        const __half* k_ = K + b * kK * kN;
        const __half* v_ = V + b * kN * kP;
        __half* o_ = O + b * kM * kP;

        // Compute attention scores: Q * K^T
        for (int i = 0; i < kM; ++i) {
            for (int j = 0; j < kN; ++j) {
                __half s = 0.;
                for (int k = 0; k < kK; ++k) {
                    s += q_[i * kK + k] * k_[k + kK * j];
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
                    s += acc[i * kN + k] * v_[k + kN * j];
                }
                exp_values[i * kP + j] = s;
            }
        }

#ifdef DEBUG
        printf("Print exp_values: \n");
        for (int i = 0; i < kM; ++i) {
            for (int j = 0; j < kP; ++j) {
                printf("%.3f ", __half2float(exp_values[i * kP + j]));
            }
            printf("\n");
        }
#endif

#ifdef DEBUG
        printf("Print new_sums: \n");
        for (int i = 0; i < kM; ++i) {
            printf("%.3f ", __half2float(new_sums[i]));
        }
        printf("\n");
#endif

#ifdef DEBUG
        printf("Print new norm vec: \n");
        for (int i = 0; i < kM; ++i) {
            printf("%.3f ", __half2float(new_norm_vec[i]));
        }
#endif

        // Compute O = (O * prev_sums * prev_norm_vec + new_norm_vec *
        // exp_values) / new_sums.
        for (int i = 0; i < kM; ++i) {
            for (int j = 0; j < kP; ++j) {
                o_[i * kP + j] =
                    (o_[i * kP + j] * prev_sums[i] * prev_norm_vec[i] +
                     new_norm_vec[i] * exp_values[i * kP + j]) /
                    new_sums[i];
            }
        }
    }
}
