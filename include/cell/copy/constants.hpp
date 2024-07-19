#pragma once

#include "config.hpp"

namespace tiledcuda::cell::copy {

enum class CopyInst {
    kLoadMat = 0,   // ldmatrix for loading data from shared memory to register.
    kStoreMat = 1,  // stmatrix for storing data from register to shared memory.
    kLoaddShared32 = 2,   // ldsm32 for loading 32-bit data from shared memory.
    kLoaddShared128 = 3,  // ldsm128 for loading 128-bit data
                          // from shared memory.
    kStoreShared32 = 4,   // stsm32 for storing 32-bit data to shared memory.
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
