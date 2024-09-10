#pragma once

#include "config.hpp"

namespace tiledcuda::cell::copy {

enum class CopyInst {
    kLoadMat = 0,   // ldmatrix for loading data from shared memory to register.
    kStoreMat = 1,  // stmatrix for storing data from register to shared memory.
    kLoadShared32 = 2,  // ldsm32 for loading 32-bit data from shared memory.
    kLoadShared128 = 3  // ldsm128 for loading 128-bit data from shared memory.
};

enum class WarpReuse {
    // TODO(haruhi): It seems that Cir/RowReuseCir/ColReuseCir are not ncessary,
    // thus the reuse mode can be simplified.
    // data are evenly partitioned to be loaded by warps.
    kCont = 0,          // all warps continuously load data, no reuse
    kCir = 1,           // all warps circularly load data, no reuse
    kRowReuseCont = 2,  // Row-wise even reuse, warps in the same row
                        // repeatedly load the same data
    kRowReuseCir = 3,   // Row-wise circular reuse
    kColReuseCont = 4,  // Column-wise even reuse, warps in the same column
                        // repeatedly load the same data
    kColReuseCir = 5    // Column-wise circular reuse
};

}  // namespace tiledcuda::cell::copy
