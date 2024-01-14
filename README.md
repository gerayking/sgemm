### Introduction
SGEMM will utilize CUDA to complete a GEMM operation. It will be optimized by following these steps:

+ Shared memory optimization (done)
+ Register file optimization (done)
+ Prefetching (in progress)
+ Bank conflict resolution
+ Fast Fused Multiply-Add (FFMA)
+ Additional optimizations to be determined...
## QuickStar
```shell
git clone https://github.com/gerayking/sgemm.git
cd sgemm
nvcc -lcublas -o test matrixMul.cu
./test
```