# TiledCUDA

## Introduction

**TiledCUDA** is a kernel template library that is designed to be highly efficient and easy to use. 

TiledCUDA takes a bottom-up approach, building efficient kernel components centered around **Base Tiles**.These components encapsulate performance-related parameters that users can configure according to their specific requrements.

The features of TiledCUDA include:

- **High-level API**: TiledCUDA provides a high-level API that is easy to use and understand.
- **Flexible**: TiledCUDA allows users to configure the kernel components according to their specific requirements.
- **Efficient**: TiledCUDA provides efficient implementations of the kernel components.

## Design

TiledCUDA builds kernels around the core concept of BaseTile, starting from the lowest level by encapsulating atomic instructions(`ldmatrix`, `stmatrix`, `mma`, etc.), and then composing them step-by-step in both the temporal and spatial domains.

To facilitate user-friendliness, TiledCUDA has implemented the **TileIterator**, which overloads the indexing operator and iterator, allowing users to control the traversal and execution of Tiles using more precise semantics.

Within the `BaseTile`, TiledCUDA defines the minimum shape that can be executed by the hardware, and provides implementations based on different data types.

![](docs/_static/TiledCUDA_overview.png)

## Examples

Here's an example of a simple GEMM kernel written in TiledCUDA:

```cpp
template <typename InType, typename AccType, typename IteratorA, typename RegA,
          typename LoaderA, typename IteratorB, typename RegB, typename LoaderB,
          typename GlobalC, typename RegC, typename CStorer>
__global__ void simple_gemm(const InType* dA, const InType* dB, AccType* dC) {
    IteratorA gAs(dA);
    RegA rA;
    LoaderA loader_a;

    IteratorB gBs(dB);
    RegB rB;
    LoaderB loader_b;

    RegC acc;

    for (int k = 0; k < IteratorA::sc1; ++k) {
        loader_a(gAs(k), rA);
        loader_b(gBs(k), rB);
        __syncthreads();

        compute::gemm_(rA, rB, acc);
    }
    __syncthreads();

    GlobalC gC(dC);
    CStorer storer_c;
    storer_c(acc, gC);
}
```

## Quick Start

### Download

```bash
git clone git@github.com:TiledTensor/TiledCUDA.git
cd TiledCUDA && git submodule update --init --recursive
```

### Installation

TileCUDA requires a C++20 host compiler, CUDA 12.0 or later, and GCC version 10.0 or higher to support C++20 features.

### Unit Test

- **Run a single unit test**: `make unit_test UNIT_TEST=test_scatter_nd.py`
- **Run all unit tests**: `./scripts/unittests/python.sh`
- **Run a single cpp unit test**: `make unit_test_cpp CPP_UT=test_copy`
- **Run all cpp unit tests**: `make unit_test_cpps`




