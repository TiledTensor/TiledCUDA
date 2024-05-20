# Data Transfer between Memory Hierarchies

## Data transfer between shared memory and resiger file

Ampere GPUs have the ability to move two-dimensional data from shared memory to threads' local register file with a single instruction. Specifically, the `ldmatrix` instruction uses a warp to move up to four $8 \times 8$ matrices, but it requires a strict data-to-thread mapping. The macro kernel `copy_2d_tile_s2r` is to efficiently feed data to the tensor core using this warp cooperative instruction, without exposing low-level complexities.

### Goals and Non-goals

**Goals**:

1. Provide a clear and understandable set of configurations.
   1. Maintain a clear separation between the declaration of ***a copy plan***, and the execution of the ***copy kernel***.
   2. Once the configuration is established, how a single thread accesses data from shared memory to its local register is fully determined.

2. Implement an clean array indexing-like syntax to enable easy access and manipulation of data tiles in the memory hierarchy by a potential code emitter.

**Non-Goals**:

1. Generalizing to non-tensor core instructions is not a priority.

### Interfaces

<p align="center">
<img src="figures/data_transfer_between_shared_and_register.png" width=80%><br>
Fig. Data transfer between shared memory and register file using ldmatrix.
</p>

1. **Declare a copy plan**
 
    ```cpp
    using Shared = SharedTile<Element, TemporalExecShared, WarpLayout,
                              ThreadLayout, ElementaryDataTileShared>;
    using Reg = RegTile<Element, TemporalExecReg, ElementaryDataTileReg>;
    ```

    Configure the copy plan according to the figure shown above:

    ```cpp
    // how many times a spatial CTA tile are executed in time
    // along the two dimensions
    using TemporalExecShared = TileShape<2, 2>;

    // configurated by a potential internal tunner
    // how warps are laied out in a CTA
    using WarpLayout = TileShape<1, 4>;
    // how threads are laid out in a single warp.
    // this configuration is fixed when using ldmatrix.
    using ThreadLayout = TileShape<16, 2>;
    // the shape of an elementary data file for a single thread.
    using ElemDataTile = TileShape<2, 16>;

    using Shared = SharedTile<Element, TemporalExecShared, WarpLayout,
                              ThreadLayout, ElemDataTile>;

    // for register tile
    // how many times an atomic instruction are executed in time
    // along the two dimensions
    using TemporalExecReg = TileShape<2, 1>;
    using ElemDataTileReg = TileShape<1, 8>;
    using Reg = RegTile<Element, TemporalExecReg, ElemDataTileReg>;
    ```

1. **Issue the copy macro kernel**

    ```cpp
    // raw shared memory data pointer
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);

    // descriptor for shared memory decomposition
    Shared s_tiles(buf);
    // descriptor for register tile
    Reg r_tile;
    // temporal execution of the macro kernel
    for (auto i = 0; i < 2; ++i) {
        for (auto j = 0; j < 2; ++j) {
            copy::copy_2d_tile_s2r(s_tiles[make_int2(i, j)] /*source tile*/,
                                   r_tile /*destinationn tile*/);
        }
    }
    ```
