#pragma once

#include "cell/traits/base.hpp"
#include "types/common.hpp"

namespace tiledcuda::cell {

namespace tl = tile_layout;

namespace {
template <typename Element, const int kRows, const int kCols, const int kStride>
__device__ void print_tile(const Element* data) {
    const int delimeter = 64;

    half* data_ = reinterpret_cast<half*>(const_cast<Element*>(data));

    if (threadIdx.x || blockIdx.x || blockIdx.y) return;

    int row, col;
    for (int i = 0; i < kRows * kCols; ++i) {
        row = i / kCols;
        col = i % kCols;

        printf("%.0f, ", __half2float(data_[row * kStride + col]));

        if ((i + 1) % delimeter == 0) printf("\n");
    }
    printf("\n\n");
}
}  // namespace

/*
 * @brief The shared memory tile is comprised of a 2-dimensional array of data
 *        elements.
 * @tparam Element_: the data type of the elements.
 * @tparam TemporalExec_: the macro kernel `copy_2d_tile_s2r` loads data from
 *         shared memory to register file, and it is executed in a temporal
 *         fashion along two dimensions. The first element of TemporalExec_ is
 *         the number of stripes along the outer dimension, and the second
 *         element is the number of stripes along the inner dimension.
 * @tparam WarpLayout_: the warps in a thread block are laid out.
 * @tparam ThreadLayout_: how threads in a warp are laied out. When using
 *         ldmatrix, `ThreadLayout_` should be fixed to `TileShape<16, 2>`.
 * @tparam DataLayout_: the data tile layout for executing the macro
 *         kernel `copy_2d_tile_s2r` for a time.
 */
template <typename Element_, typename TemporalExecShape_, typename WarpLayout_,
          typename ThreadLayout_, typename ElemTileLayout_,
          typename Base = traits::TraitsBase<Element_>>
class SharedTile : public Base {
  public:
    using DType = Element_;
    using TemporalExec = TemporalExecShape_;

    using WarpLayout = WarpLayout_;
    using ThreadLayout = ThreadLayout_;
    using ElemTileLayout = ElemTileLayout_;

    // Shared memory can be accessed by all threads within a thread block.
    // It's necessary to keep track of the number of threads in the block to
    // ensure proper access to shared memory.
    constexpr static int kThreads = tl::get_numel<WarpLayout> * 32;

    // The shared memory tile has a shape of [kRows, kCols] is made up of
    // elementary data tile, each of which is accessed by a single thread.

    constexpr static int kRowsPerExec = tl::num_rows<ElemTileLayout> *
                                        tl::num_rows<ThreadLayout> *
                                        tl::num_rows<WarpLayout>;
    constexpr static int kColsPerExec = tl::num_cols<ElemTileLayout> *
                                        tl::num_cols<ThreadLayout> *
                                        tl::num_cols<WarpLayout>;

    constexpr static int kRows = kRowsPerExec * dim_size<0, TemporalExec>;
    constexpr static int kCols = kColsPerExec * dim_size<1, TemporalExec>;
    constexpr static int kNumel = kRows * kCols;

    // stripe counts for temporal execution of the copy plan
    constexpr static size_t sc0 = dim_size<0, TemporalExec>;
    constexpr static size_t sc1 = dim_size<1, TemporalExec>;

    // FIXME: The code below that determines whether the register tile is
    // created by ldmatrix or wmma is quite tricky. These two instructions leads
    // to different shared memory access patterns. It's important to fix this
    // tricky implementations to ensure that the code is more readable and
    // easier to work with.
    constexpr static bool kIsWmmaTile =
        tl::num_rows<ThreadLayout_> == 8 && tl::num_cols<ThreadLayout_> == 4;

    // NOTE that: this implementation assumes that each thread uses the maximal
    // width of vectorized access, which we treat as an atomic access
    // instruction, to optimize performance on the GPU platform. Specifically,
    // we use 1 x 128 bits per access. Given the shape of the elementary tile
    // (denoted by `DataLayout`), `PtrArray` stores the number of times and the
    // positions in shared memory where the atomic memory access instruction is
    // executed.
    constexpr static int kPtrNum =
        tl::get_numel<ElemTileLayout> / Base::kNumPerAccess;

    // FIXME: hard-coded for wmma
    constexpr static int kNumPerStore = 2;
    constexpr static int kPtrNumWmma =
        tl::get_numel<ElemTileLayout> / kNumPerStore;
    using PtrArray = DevArray<DType*, kIsWmmaTile ? kPtrNumWmma : kPtrNum>;

  public:
    // ctor
    DEVICE SharedTile(DType* data) : data_(data), shm_pos_(PtrArray{}) {}

    DEVICE DType* mutable_data() { return data_; }

    DEVICE const DType* data() const { return data_; }

    // @brief compute the shared memory positions accessed by the current thread
    //        (referred to `threadIdx.x`, this assumes the implementation
    //        uses 1-D thread indexing) in composing the spatial shared memory
    //        tile.
    // @tparam idx: the 2D coordinates of the temporal execution of a spatial
    //              shared memory tile.
    DEVICE PtrArray& operator[](int2 idx) {
        if (kIsWmmaTile) {
            compute_shm_pos_wmma(idx.x, idx.y);
        } else {
            compute_shm_pos_ldmatrix(idx.x, idx.y);
        }

        return shm_pos_;
    }

    DEVICE void operator()(int x, int y) {}

  private:
    DEVICE void compute_shm_pos_ldmatrix(int x, int y) {
        // TODO(haruhi): the current implementation assumes shared memory
        // columns are contiguous in memory.
        DType* data = data_;

        // step 1. advance pointer to a all-warps-tile:
        int stride = kCols * kRowsPerExec;
        int col_stride = kColsPerExec;
        int offset = x * stride + y * col_stride;
        data += offset;  // advance pointer

        // step 2. advance the pointer to the shared memory position for the
        // current warp indexed by: [warp_row, warp_col].
        // compute the 2D coordinates of the warp that the current thread
        // belongs to.
        int warp_id = threadIdx.x / 32;  // 1-D warp index
        int warp_row = warp_id / tl::num_cols<WarpLayout>;
        int warp_col = warp_id % tl::num_cols<WarpLayout>;

        int row_stride = kCols * tl::num_rows<ThreadLayout>;
        col_stride = Base::kNumPerAccess * tl::num_cols<ThreadLayout>;
        offset = warp_row * row_stride + warp_col * col_stride;
        data += offset;  // advance pointer

        // step 3. advance the pointer to shared memory position the current
        // thread inside a warp cooperative instruction (indexed by
        // [lane_row, lane_col]) access. compute the 2D coordinates
        int lane_id = threadIdx.x % 32;
        // in a warp cooperative instruction, the threads are
        // laid out in a [2, 2] tile, and each tile has a shape of [8, 1].
        int n = lane_id / 8;
        int lane_row = n / 2 * 8 + lane_id - n * 8;
        int lane_col = n % 2;
        col_stride = Base::kAccessInBits / 8 / sizeof(DType);
        offset = lane_row * kCols + lane_col * col_stride;
        data += offset;  // advance pointer

        row_stride =
            tl::num_rows<ThreadLayout> * tl::num_rows<WarpLayout> * kCols;
        col_stride = Base::kNumPerAccess * tl::num_cols<ThreadLayout> *
                     tl::num_cols<WarpLayout>;
        int sc0 = tl::num_rows<ElemTileLayout>;
        int sc1 = tl::num_cols<ElemTileLayout> / Base::kNumPerAccess;

        for (int i = 0; i < sc0; ++i) {
            for (int j = 0; j < sc1; ++j)
                shm_pos_[i * sc1 + j] = data + j * col_stride;
            data += row_stride;
        }
    }

    DEVICE void compute_shm_pos_wmma(int x, int y) {
        DType* data = data_;

        int lane_id = threadIdx.x % 32;
        int lane_row = lane_id / 4;
        int lane_col = lane_id % 4;

        int col_stride = tl::num_cols<ElemTileLayout> * 2;
        int row_stride = tl::num_cols<ElemTileLayout> *
                         tl::num_cols<ThreadLayout> *
                         tl::num_rows<ThreadLayout>;

        int row, col, offset;
        for (int i = 0; i < PtrArray::kSize; i += 4) {
            for (int j = 0; j < 4; ++j) {
                // hard-coded the inner thread layout for wmma
                row = j % 2;
                col = j / 2;
                offset = row * row_stride + col * col_stride + lane_row * 16 +
                         lane_col * 2;

                shm_pos_[i * 4 + j] = data + offset;
            }
        }
    }

    DType* data_;  // Pointer to shared memory data.
    // shared memory positions accessed by the current thread.
    PtrArray shm_pos_;
};

template <class Element_, class TileLayout_>
class Shared {
  public:
    using DType = Element_;
    using Layout = TileLayout_;

    static constexpr int kNumel = tl::get_numel<Layout>;
    static constexpr int kRows = tl::num_rows<Layout>;
    static constexpr int kCols = tl::num_cols<Layout>;

    DEVICE Shared(DType* data) : data_(data) {}

  private:
    DType* data_;
};

}  // namespace tiledcuda::cell
