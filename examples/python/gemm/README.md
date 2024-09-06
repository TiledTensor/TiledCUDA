This example demonstrates how to use TiledCuda's macro kernels to compose a simple GEMM and auto-tune some performance-critical parameters.

In this simple GEMM implementation, data tiles are loaded from global memory directly into a thread's local registers. TensorCore's WMMA is then used to compute GEMM on the registers, and finally, the results are stored back to global memory.

> [!Note]
> *This example is for demonstration purposes and does not leverage shared memory.*

To execute the example, run:

```bash
python3 main.py 2>&1 | tee log.tsv
```
