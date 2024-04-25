# TiledCUDA

## Introduction

**TiledCUDA** is an efficient kernel template library written in **CuTe**, which provides a wrapper for cutlass CuTe and enables more efficient fusion.

TiledCUDA uses **PyTorch** as the runtime and leverages the **Tensor** class of PyTorch for convenient testing.

## Quick Start

### Download

```bash
git clone git@github.com:TiledTensor/TiledCUDA.git
cd TiledCUDA && git submodule update --init --recursive
```

### Unit Test

- **Run single unit test**: `make unit_test UNIT_TEST=test_scatter_nd.py`
- **Run all unit tests**: `./scripts/unittests/python.sh`

## Features

- Implemented `__device__` function wrapper that enables **static/dynamic** copying between different memory hierarchy.
- Implemented `__device__` function wrapper for CUDA **micro kernels**, such as `copy_async` and tensor core operations.
- Implemented template wrapper for **CuTe** to make it easier to use.
- Implemented fused kernels such as **GEMM**, **Back2Back GEMM**, **Batched GEMM**, **Lstm Cell**, etc.
