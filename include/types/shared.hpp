#pragma once

namespace tiledcuda {

template <typename Element_, typename DataLayout_, typename ThreadLayout_>
class SharedTile {
  public:
    using Element = Element_;
    using DataLayout = DataLayout_;
    using ThreadLayout = ThreadLayout_;

    DEVICE SharedTile(Element* data)
        : data_(data), layout_(DataLayout{}), t_layout_(ThreadLayout{}){};

    const static int kRows = tl::num_rows<DataLayout>;
    const static int kCols = tl::num_cols<DataLayout>;

    DEVICE Element* operator[](size_t idx) { return &data_[layout_(idx, 0)]; }

    DEVICE const Element* operator[](size_t idx) const {
        return &data_[layout_(idx, 0)];
    }

    DEVICE Element* operator[](int2 idx) {
        return &data_[layout_(idx.x, idx.y)];
    }

    DEVICE const Element* operator[](int2 idx) const {
        return &data_[layout_(idx.x, idx.y)];
    }

    DEVICE int count() const { return 1; }

    DEVICE Element* mutable_data() { return data_; }

    DEVICE const Element* data() const { return data_; }

  private:
    DataLayout layout_;
    ThreadLayout t_layout_;
    Element* data_;  // shared memory tile data
};

}  // namespace tiledcuda
