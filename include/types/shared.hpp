#pragma once

namespace tiledcuda {

template <typename Element_, typename Layout_>
class SharedTile {
  public:
    using Element = Element_;
    using Layout = Layout_;

    DEVICE SharedTile(Element* data) : data_(data), layout_(Layout{}){};

    const static int kRows = tl::num_rows<Layout>;
    const static int kCols = tl::num_cols<Layout>;

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

    DEVICE Element* mutable_data() { return data_; }

    DEVICE const Element* data() const { return data_; }

  private:
    Layout layout_;
    Element* data_;  // shared memory tile data
};

}  // namespace tiledcuda
