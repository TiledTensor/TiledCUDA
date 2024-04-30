# TiledCUDA

## Introduction

**TiledCUDA** is a kernel template library that is designed to be highly efficient. It provides a wrapper for cutlass **CuTe** to simplifly the process of implementing complex fused kernels that utilize tensor core GEMM.

TiledCUDA utilizes **PyTorch** as its runtime environment and leverages the **Tensor** class of PyTorch for convenient testing.

## Quick Start

### Download

```bash
git clone git@github.com:TiledTensor/TiledCUDA.git
cd TiledCUDA && git submodule update --init --recursive
```

### Unit Test

- **Run a single unit test**: `make unit_test UNIT_TEST=test_scatter_nd.py`
- **Run all unit tests**: `./scripts/unittests/python.sh`
- **Run a single cpp unit test**: `make unit_test_cpp CPP_UT=test_copy`
- **Run all cpp unit tests**: `make unit_test_cpps`

## Features

- Implemented `__device__` function wrapper that enables **static/dynamic** copying between different memory hierarchy.
- Implemented `__device__` function wrapper for CUDA **micro kernels**, such as `copy_async` and tensor core operations.
- Implemented template wrapper for **CuTe** to simplify its usage.
- Implemented fused kernels such as **GEMM**, **Back2Back GEMM**, **Batched GEMM**, **Lstm Cell**, etc.
