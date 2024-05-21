#pragma once

namespace tiledcuda::cell {

/// @brief: A fixed-length small 1D array on the device.
/// usage: store a set of data pointers on device.
template <typename Element_, const int N_>
class DevArray {
  public:
    using Element = Element_;
    constexpr static int kSize = N_;

    DEVICE DevArray() { memset((void*)data_, 0, sizeof(data_)); }

    DEVICE Element& operator[](int idx) { return data_[idx]; }

  private:
    Element data_[kSize];
};

}  // namespace tiledcuda::cell
