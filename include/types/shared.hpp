#pragma once

#include "types/common.hpp"
#include "types/tile_shape.hpp"

namespace tiledcuda::cell {

template <typename Element_, typename TemporalExec_, typename WarpLayout_,
          typename ThreadLayout_, typename DataLayout_>
class SharedTile {
  public:
    using DType = Element_;
    using TemporalExec = TemporalExec_;
    using WarpLayout = WarpLayout_;
    using ThreadLayout = ThreadLayout_;
    using DataLayout = DataLayout_;

    constexpr static int kThreads = get_numel<WarpLayout> * 32;

    constexpr static int kRows =
        dim_size<0, DataLayout> * dim_size<0, ThreadLayout> *
        dim_size<0, WarpLayout> * dim_size<0, TemporalExec>;
    constexpr static int kCols =
        dim_size<1, DataLayout> * dim_size<1, ThreadLayout> *
        dim_size<1, WarpLayout> * dim_size<1, TemporalExec>;
    constexpr static int kNumel = kRows * kCols;

    static constexpr int kElemPerThread = get_numel<DataLayout>;
    using ShmPtr = DevArray<DType, kElemPerThread>;

    // ctor
    DEVICE SharedTile(DType* data) : data_(data){};

    DEVICE ShmPtr& operator[](int2 idx) {
        // FIXME(haruhi): Not implemented yet. Compelete the implementation.
        return shm_ptrs_[idx.x + idx.y];
    }

    DEVICE const DType* operator[](int2 idx) const {
        // FIXME(haruhi): Not implemented yet. Compelete the implementation.
        return static_cast<DType*>(data_);
    }

    DEVICE DType* mutable_data() { return data_; }

    DEVICE const DType* data() const { return data_; }

  private:
    DType* data_;  // shared memory tile data

    // An array of pointers that records the shared memory positions the the
    // current thread (indexed by thread.idx).
    ShmPtr* shm_ptrs_;
    size_t shm_ptrs_count_;
};

}  // namespace tiledcuda::cell
