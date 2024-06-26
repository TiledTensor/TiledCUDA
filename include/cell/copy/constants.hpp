#pragma once

#include "config.hpp"

namespace tiledcuda::cell::copy {

enum class CopyInst {
    LoadMat = 0,   // ldmatrix for loading data from shared memory to register.
    StoreMat = 1,  // stmatrix for storing data from register to shared memory.
    LoadS32 = 2,   // ldsm32 for loading 32-bit data from shared memory.
    LoadS128 = 3   // ldsm128 for loading 128-bit data from shared memory.
};

// FIXME(haruhi): A quick implementation. Consider whether this enum should be
// coupled with RegTile.
enum class RegLayout {
    FMA = 0,  // Tile layout for FMA.

    // Refer to this slide for details on how data is distributed in each
    // thread's local register after the TCU's WMMA operation:
    // https://developer.download.nvidia.com/video/gputechconf/gtc/2020/
    // presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf
    // WMMA shape is "m16n16k16
    WMMA_m16n16k16 = 1,
};

enum class WarpReuse {
    // TODO(haruhi): It seems that Cir/RowReuseCir/ColReuseCir are not ncessary,
    // thus the reuse mode can be simplified.
    // data are evenly partitioned to be loaded by warps.
    Cont = 0,          // all warps continuously load data, no reuse
    Cir = 1,           // all warps circularly load data, no reuse
    RowReuseCont = 2,  // Row-wise even reuse, warps in the same row
                       // repeatedly load the same data
    RowReuseCir = 3,   // Row-wise circular reuse
    ColReuseCont = 4,  // Column-wise even reuse, warps in the same column
                       // repeatedly load the same data
    ColReuseCir = 5    // Column-wise circular reuse
};

}  // namespace tiledcuda::cell::copy
