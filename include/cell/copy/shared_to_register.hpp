#pragma once

#include "cell/copy/constants.hpp"
#include "types/mod.hpp"

namespace tiledcuda::cell::copy {

namespace detail {

// functor to copy data from shared memory to register file.
template <typename Shared, typename Reg, typename WarpLayout,
          CopyInst kCopyInst>
struct CopyShared2Reg {
    DEVICE void operator()(const Shared& src, Reg& dst, WarpReuse kMode);
};

// partial specialization for ldmatrix
template <typename Shared, typename Reg, typename WarpLayout>
struct CopyShared2Reg<Shared, Reg, WarpLayout, CopyInst::LoadMat> {
    DEVICE void operator()(const Shared& src, Reg& dst, WarpReuse kMode) {
        // implement this
    }
};

// functor to copy data from shared memory to register file.
template <typename Reg, typename Shared, typename InstShape,
          RegLayout kRegLayout, CopyInst kCopyInst>
struct CopyReg2Shared {
    DEVICE void operator()(const Reg& src, Shared& dst);
};

// partial specialization for wmma 16x16x16, and LDSM32
template <typename Reg, typename Shared>
struct CopyReg2Shared<Reg, Shared, InstShape<16, 16, 16>, RegLayout::TileWMMA,
                      CopyInst::LoadS32> {
    DEVICE void operator()(const Reg& src, Shared& dst) {}
};
}  // namespace detail

/// a warper function for the situation that `Shared` are computed from some
/// runtime value.
template <typename Shared, typename Reg, typename WarpLayout>
DEVICE void copy_tile_s2r(const Shared& src, Reg& dst, const WarpLayout& layout,
                          WarpReuse kMode) {
    using Copy =
        detail::CopyShared2Reg<Shared, Reg, WarpLayout, CopyInst::LoadMat>;

    Copy copy;
    copy(src, dst, kMode);
}

template <typename Reg, typename Shared>
DEVICE void copy_tile_r2s(const Reg& src, Shared& dst) {
    using Copy = detail::CopyReg2Shared<Reg, Shared, InstShape<16, 16, 16>,
                                        RegLayout::TileWMMA, CopyInst::LoadS32>;
    Copy copy;
    copy(src, dst);
}

}  // namespace tiledcuda::cell::copy
