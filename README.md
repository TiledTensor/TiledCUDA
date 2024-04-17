# TiledCUDA

## Introduction

**TiledCUDA** is a kernel library implemented with CuTe. Its goal is to achieve more efficient fusion strategies.


## Usage
### Download
```
git clone git@github.com:TiledTensor/TiledCUDA.git
cd TiledCUDA && git submodule update --init --recursive
```

### Unit Test
```
make unit_test UNIT_TEST=test_scatter_nd.py
```