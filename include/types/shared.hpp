#pragma once

namespace tiledcuda {

template <typename Element_, typename Layout_>
struct SharedTile {
    using Element = Element_;
    using Layout = Layout_;

   public:
    DEVICE SharedTile(Element* data) : data_(data) {}

    DEVICE Element* operator[](size_t idx) { return nullptr; }

    DEVICE const Element* operator[](size_t idx) const { return nullptr; }

    // DEVICE Element& operator[](int2 idx) { return data_[idx.x][idx.y]; }

    // DEVICE const Element& operator[](int2 idx) const {
    //     return data_[idx.x][idx.y];
    // }

   private:
    Element* data_;  // shared memory tile data
};

}  // namespace tiledcuda
