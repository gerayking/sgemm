#include <stdio.h>
#include <stdlib.h>
#include "assert.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>


#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}


template<
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N,
    const int THREAD_SIZE_X,
    const int THREAD_SIZE_Y
>
__global__ void Sgemm(
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C,
    const int M,
    const int N,
    const int K){
        int bx = blockIdx.x;
        int by = blockIdx.y;

        int tx = threadIdx.x;
        int ty = threadIdx.y;


        const int THREAD_X_PER_BLOCK = blockDim.x;
        const int THREAD_Y_PER_BLOCK = blockDim.y;
        const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

        const int tid = ty * THREAD_X_PER_BLOCK + tx;

        //
        __shared__ float As[BLOCK_SIZE_K][BLOCK_SIZE_M];
        __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];

        float transfer_A[BLOCK_SIZE_K];

        const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4; // 8 / 4 = 2
        const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4; // 128 / 4 = 32

        const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW; // 256 / 2
        const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW; // 256 / 32


        const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
        const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

        const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
        const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;
        A = &A[(BLOCK_SIZE_M * by) * K];
        B = &B[BLOCK_SIZE_N*bx];

        float frag_a[BLOCK_SIZE_K];
        float frag_b[BLOCK_SIZE_K];

        float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};

        int tile_idx = 0;
        do
        {
            // load content from global memory to shared memory
            for(int i=0;i<BLOCK_SIZE_M;i+=A_TILE_ROW_STRIDE){
                int ldg_a_index = i / A_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(transfer_A[ldg_a_index]) = FETCH_FLOAT4(A[OFFSET(
                    A_TILE_ROW_START + i,
                    A_TILE_COL + tile_idx,
                    K
                )]);
                // printf("%f %f %f %f\n",transfer_A[ldg_a_index],transfer_A[ldg_a_index+1],transfer_A[ldg_a_index+2],transfer_A[ldg_a_index+3]);
                As[A_TILE_COL][A_TILE_ROW_START + i] = transfer_A[ldg_a_index];
                As[A_TILE_COL + 1][A_TILE_ROW_START + i] = transfer_A[ldg_a_index + 1];
                As[A_TILE_COL + 2][A_TILE_ROW_START + i] = transfer_A[ldg_a_index + 2];
                As[A_TILE_COL + 3][A_TILE_ROW_START + i] = transfer_A[ldg_a_index + 3];
            }
            for(int i=0;i<BLOCK_SIZE_K;i+=B_TILE_ROW_STRIDE){
                FETCH_FLOAT4(Bs[B_TILE_ROW_START+i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                    tile_idx + B_TILE_ROW_START + i,
                    B_TILE_COL,
                    N
                )]);
            }
            __syncthreads();
            for(int j=0;j<BLOCK_SIZE_K;j++){
                for(int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4){
                    FETCH_FLOAT4(frag_a[thread_y]) = FETCH_FLOAT4(As[j][THREAD_SIZE_Y * ty + thread_y]);
                }
                for(int thread_x =0; thread_x < THREAD_SIZE_X; thread_x += 4){
                    FETCH_FLOAT4(frag_b[thread_x]) = FETCH_FLOAT4(Bs[j][THREAD_SIZE_X * tx + thread_x]);
                }
                for(int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y){
                    for(int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x){
                        accum[thread_y][thread_x] += frag_a[thread_y] * frag_b[thread_x];
                    }
                }
            }
            tile_idx += BLOCK_SIZE_K;
        } while (tile_idx < K);

        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x+=4) {
                FETCH_FLOAT4(C[OFFSET(
                    BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                    BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                    N)]) = FETCH_FLOAT4(accum[thread_y][thread_x]);
            }
        }
    }

    int main(int argc, char** argv) {
    if (argc != 4) {
        printf("usage: ./main [M] [K] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);

    assert( M%8 == 0); 
    assert( N%8 == 0); 
    assert( K%8 == 0); 

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;
    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)malloc(bytes_C);
    float* h_C1 = (float*)malloc(bytes_C);

    float* d_A;
    float* d_B;
    float* d_C;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));
    checkCudaErrors(cudaMalloc(&d_C, bytes_C));
    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = 2.0 * M * N * K;

    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;

    // generate A
    for( int i = 0; i < M * K; i++ ){
        h_A[i] = i / 13;
    }

    // generate B
    for( int i = 0; i < K * N; i++ ) {
        h_B[i] =  i / 13;
    }

    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 1000;

    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
        Sgemm<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, M, N, K);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));


    checkCudaErrors(cudaMemcpy( h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[0] = msecTotal / nIter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[0],
        msecPerMatrixMul[0],
        flopsPerMatrixMul);

    // cublas
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
            M, N, K, &alpha, 
            d_A, K, d_B, N, &beta, d_C, N
        );
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[1] = msecTotal / nIter;
    gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
    printf( "CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[1],
        msecPerMatrixMul[1],
        flopsPerMatrixMul);

    cublasDestroy(blas_handle); 
    
    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        int row = i / N;
        int col = i % N;
        double abs_err = fabs(h_C[i] - h_C1[col * M + row]);
        double dot_length = M;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, h_C[i], h_C1[col * M + row], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);
    
    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C1);
}
