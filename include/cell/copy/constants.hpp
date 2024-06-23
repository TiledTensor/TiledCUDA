#pragma once

namespace tiledcuda::cell::copy {

enum class CopyInst {
    LoadMat = 0,   // ldmatrix for loading data from shared memory to register.
    StoreMat = 1,  // stmatrix for storing data from register to shared memory.
    LoadS32 = 2,   // ldsm32 for loading 32-bit data from shared memory.
    LoadS128 = 3   // ldsm128 for loading 128-bit data from shared memory.
};

enum class RegLayout {
    TileWMMA = 0,  // Tile layout for TCU WMMA.
};

enum class WarpReuse {
    NoReuse = 0,  // No reuse.
    Cont = 1,     // Continuous
    Cir = 2,      // Circular
    RowCont = 3,  // RowWiseContinuouslyReuse
    RowCir = 4,   // RowWiseCircularlyReuse
    ColCont = 5,  // ColWiseContinuouslyReuse
    ColCir = 6    // ColWiseCircularlyReuse
};

}  // namespace tiledcuda::cell::copy
