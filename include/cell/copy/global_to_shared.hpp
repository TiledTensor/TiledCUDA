#pragma once

#include "cell/copy/mod.hpp"
#include "cell/copy/warp.hpp"
#include "cell/traits/base.hpp"
#include "types/mod.hpp"

namespace tiledcuda::cell::copy {

using namespace traits;
namespace tl = tile_layout;

template <typename Global_, typename Shared_, const int kRowExec_,
          const int kColExec_, const tl::Layout kType>
struct GlobalToRegLoaderImpl {
    using Global = Global_;
    using Shared = Shared_;
    using DType = typename Global::DType;

    static const int kRowExec = kRowExec_;
    static const int kColExec = kColExec_;

    DEVICE void operator()(const DType* src, DType* dst);
};

template <typename Global_, typename Shared_, const int kRowExec_,
          const int kColExec_>
struct GlobalToRegLoaderImpl<Global_, Shared_, kRowExec_, kColExec_,
                             tl::Layout::kSwizzledRowMajor> {
    using Global = Global_;
    using Shared = Shared_;
    using DType = typename Global::DType;

    static const int kRowExec = kRowExec_;
    static const int kColExec = kColExec_;

    DEVICE void operator()(const DType* src, DType* dst) {}
};

template <typename Global_, typename Shared_, typename WarpLayout_>
struct GlobalToRegLoader {
    using Global = Global_;
    using Shared = Shared_;
    using DType = typename Global::DType;

    using WarpLayout = WarpLayout_;

    DEVICE operator(const Global& src, Shared& dst);
};

}  // namespace tiledcuda::cell::copy