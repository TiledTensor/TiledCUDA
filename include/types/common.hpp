#pragma once

namespace tiledcuda {

/// @brief: A fixed-length small array on the device.
/// usage: store a set of data pointers on device.
template <typename Element_, const int N_>
class DevArray {
    using Element = Element_;
    constexpr static int N = N_;

  public:
    DEVICE Element& operator[](int idx) {
        if (idx < 0 || idx >= N) {
            // FIXME(haruhi): throw an exception.
        }
        return data_[idx];
    }

  private:
    Element* data_[N];
};

}  // namespace tiledcuda
