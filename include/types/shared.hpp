#pragma once

#include "types/common.hpp"

namespace tiledcuda {

template <typename Element_, typename WholeLayout_, typename ElementLayout_,
          typename ThreadLayout_>
class SharedTile {
  public:
    using Element = Element_;
    using WholeLayout = WholeLayout_;
    using ElementLayout = ElementLayout_;
    using ThreadLayout = ThreadLayout_;

    static constexpr int kSrcNumel = size(WholeLayout{});
    static constexpr int kThreads = size(ThreadLayout{});

    static constexpr int kRows = tl::num_rows<WholeLayout>;
    static constexpr int kCols = tl::num_cols<WholeLayout>;

    // @brief DevArray stores the shared memory pointers passed to ldmatrix by a
    // single thread.
    // FIXME(haruhi): fake number as a placeholder to pass compile.
    static constexpr int N_per_threads = 4;  // number of elements per thread
    using ShmPtr = DevArray<Element, N_per_threads>;

    DEVICE SharedTile(Element* data) : data_(data), dlayout_(WholeLayout{}){};

    DEVICE ShmPtr& operator[](int2 idx) {
        // FIXME(haruhi): Not implemented yet. Compelete the implementation.
        // int tid = threadIdx.x;
        return shm_ptrs_[idx.x + idx.y];
    }

    DEVICE const Element* operator[](int2 idx) const {
        // FIXME(haruhi): Not implemented yet. Compelete the implementation.
        // int tid = threadIdx.x;
        return static_cast<Element*>(data_);
    }

    DEVICE Element* mutable_data() { return data_; }

    DEVICE const Element* data() const { return data_; }

  private:
    WholeLayout dlayout_;
    Element* data_;  // shared memory tile data

    // An array of pointers.
    ShmPtr* shm_ptrs_;
    size_t shm_ptrs_count_;
};

}  // namespace tiledcuda
