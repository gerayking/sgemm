#include <stdio.h>
#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <random>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
const int BLOCK_SIZE_WIDTH = 128;
const int BLOCK_SIZE_K = 8;
const int REGISTER_WIDTH_M = 8;
const int REGISTER_WIDTH_N = 8;
void MatrixMulHost(float* Md, float* Nd, float* Pd, int m, int n, int Width) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float Pvalue = 0;
            for (int k = 0; k < Width; ++k) {
                float Mdelement = Md[i * Width + k];
                float Ndelement = Nd[k * n + j];
                Pvalue += Mdelement * Ndelement;
            }
            Pd[i * n + j] = Pvalue;
        }
    }
}

__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int m, int n, int Width) {
    int Row = blockIdx.y * blockDim.x + threadIdx.y;
    int Col = blockIdx.x * blockDim.y + threadIdx.x;
    float Pvalue = 0;
    for (int k = 0; k < Width; ++k) {
        float Mdelement = Md[Row * Width + k];
        float Ndelement = Nd[k * Width + Col];
        Pvalue += Mdelement * Ndelement;
    }
    Pd[Row * n + Col] = Pvalue;
}
// 分块，使用__shared__ memory 来隐藏global memory -> share memory的延迟
__global__ void MatrixMulV2Kernel(float* Md, float* Nd, float* Pd, int m, int n, int Width) {

    __shared__ float Mds[BLOCK_SIZE_WIDTH][BLOCK_SIZE_K]; //block level
    __shared__ float Nds[BLOCK_SIZE_K][BLOCK_SIZE_WIDTH]; //block level
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = by * blockDim.x + ty;
    int Col =bx * blockDim.y + tx;
    float Pvalue = 0;
    for(int i = 0;i < Width/BLOCK_SIZE_K;i++){
        if(tx < BLOCK_SIZE_K){
            Mds[ty][tx] = Md[Row*Width+i*BLOCK_SIZE_K+tx];
        }
        if(ty < BLOCK_SIZE_K){
            Nds[ty][tx] = Nd[Col + (i*BLOCK_SIZE_K+ty)*Width];
        }
        __syncthreads();
        for(int k=0;k<BLOCK_SIZE_K;k++){
            float Mdelement = Mds[ty][k];
            float Ndelement = Nds[k][tx];
           Pvalue += Mdelement * Ndelement;
        }
        __syncthreads();
    }
    Pd[Row * n + Col] = Pvalue;
}

__global__ void MatrixMulV3Kernel(float* Md, float* Nd, float* Pd, int m, int n, int Width) {

    __shared__ float Mds[BLOCK_SIZE_WIDTH][BLOCK_SIZE_K]; //block level
    __shared__ float Nds[BLOCK_SIZE_K][BLOCK_SIZE_WIDTH]; //block level

    float Mdr[REGISTER_WIDTH_M];
    float Ndr[REGISTER_WIDTH_M];

    float localD[REGISTER_WIDTH_M][REGISTER_WIDTH_N];

    #pragma unroll
    for(int i=0;i<REGISTER_WIDTH_M;i++){
        for(int j=0;j<REGISTER_WIDTH_N;j++){
            localD[i][j] = 0;
        }
    }
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = (by * blockDim.y + ty)*BLOCK_SIZE_K;
    int Col = (bx * blockDim.x + tx)*BLOCK_SIZE_K;
    for(int i = 0;i < Width/BLOCK_SIZE_K;i++){
        // load gmem to smem
        if(ty==0){
            #pragma unroll
            for(int x=0;x<REGISTER_WIDTH_M;x++){
                for(int y=0;y<REGISTER_WIDTH_N;y++){
                    int dx = tx*REGISTER_WIDTH_M + x;
                    // (Row+x)*Width+i*BLOCK_SIZE_K+y
                    Mds[dx][y] = Md[(Row+dx)*Width+i*BLOCK_SIZE_K+y];
                }
            }
        }
        // Mds[ty][tx] = Md[Row*Width+i*BLOCK_SIZE_K+tx];
        if(tx==0){
            #pragma unroll
            for(int x=0;x<REGISTER_WIDTH_M;x++){             
                for(int y=0;y<REGISTER_WIDTH_N;y++){
                    int dy = ty*REGISTER_WIDTH_M+y;
                    // Col + y + (i*BLOCK_SIZE_K+x)*Width
                    Nds[x][dy] = Nd[Col + dy + (i*BLOCK_SIZE_K+x)*Width];
                }
            }
        }
        // Nds[ty][tx] = Nd[Col + (i*BLOCK_SIZE_K+ty)*Width];
        __syncthreads();
        for(int t=0;t<REGISTER_WIDTH_M;t++){
            int mOffset = ty  * REGISTER_WIDTH_M ;
            int nOffset = tx  * REGISTER_WIDTH_M ;
            // load smem from shared to register
            #pragma unroll
            for(int j = 0;j<REGISTER_WIDTH_M;j++){
                Mdr[j] = Mds[mOffset+j][t];
            }
            #pragma unroll
            for(int j=0;j<REGISTER_WIDTH_M;j++){
                Ndr[j] = Nds[t][nOffset+j];
            }
            // if(bx==0&&by==0&&tx==0&&ty==0&&i==0){
            //     // if(t==0){
            //     // for(int i=0;i<BLOCK_SIZE_WIDTH;i++){
            //     //     for(int j=0;j<BLOCK_SIZE_K;j++){
            //     //         printf("%f ",Mds[i][j]);
            //     //     }
            //     //     printf("\n");
            //     // }
            //     // for(int i=0;i<BLOCK_SIZE_K;i++){
            //     //     for(int j=0;j<BLOCK_SIZE_WIDTH;j++){
            //     //         printf("%f ",Nds[i][j]);
            //     //     }
            //     //     printf("\n");
            //     // }
            //     // }
            //     for(int i=0;i<8;i++){
            //             printf("%f ",Ndr[i]);
            //     }
            //     printf("\n");
            //     for(int i=0;i<REGISTER_WIDTH_N;i++){
            //          printf("%f ",Mdr[i]);
            //     }
            //     printf("\n");
            // }
            // computer matrix multiply accmulate 8 * 8
            #pragma unroll
            for(int j=0;j<REGISTER_WIDTH_M;j++){
                for(int k=0;k<REGISTER_WIDTH_N;k++){
                    localD[j][k] += Mdr[j] * Ndr[k];
                    // for(int m=0;m<REGISTER_WIDTH_M;m++){
                    //     localD[j][k] += Mdr[j][m]*Ndr[m][k]; 
                    //     if(bx==0&&by==0&&tx==0&&ty==0&&j==0&&k==0&&i==0){
                    //         printf("t: %d : (%d %d : %f*%f : %d %d)\n",t,j,m,Mdr[j][m],Ndr[m][k],m,j);
                    //     }
                    // }
                }
            }
        }
        // cal mul
        __syncthreads();
    }
    for(int j=0;j<REGISTER_WIDTH_M;j++){
        for(int k=0;k<REGISTER_WIDTH_N;k++){
            Pd[(j+Row)*Width+k+Col] = localD[j][k];
        }
    }
}

void testMatrixMulV3Kernel(float* Md, float* Nd, float* Pd, int m, int n, int Width){
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1); // 记录开始时间
    // 启动核函数
    const int block_size_x = 128;
    const int block_size_y = 128;
    dim3 threadsPerBlockV2(block_size_x/REGISTER_WIDTH_M, block_size_y/REGISTER_WIDTH_M); // 每个线程块包含16x16个线程
    dim3 numBlocksV2(m / block_size_x, n / block_size_y); // 根据矩阵大小设置线程块数量
    MatrixMulV3Kernel<<<numBlocksV2, threadsPerBlockV2>>>(Md, Nd, Pd, m, n, Width);
    cudaEventRecord(stop1); // 记录结束时间
    cudaEventSynchronize(stop1);
    float milliseconds_v2 = 0;
    cudaEventElapsedTime(&milliseconds_v2, start1, stop1);
    printf("Execution MatrixMulV3Kernel Time: %f ms\n", milliseconds_v2);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s ROW COL WIDTH\n", argv[0]);
        return 1; // 返回非零值表示出错
    }

    int m = atoi(argv[1]); // 将第一个参数转换为整数
    int n = atoi(argv[2]); // 将第二个参数转换为整数
    int Width = atoi(argv[3]); // 将第三个参数转换为整数

    // 分配内存并初始化输入矩阵 Md 和 Nd
    float* Md, * Nd, * Pd;
    float* hostMd, * hostNd, * hostPd, * hostAns;
    hostMd = (float*)malloc(m * Width * sizeof(float));
    hostNd = (float*)malloc(Width * n * sizeof(float));
    hostPd = (float*)malloc(m * n * sizeof(float));
    hostAns = (float*)malloc(m * n * sizeof(float));
    cudaError_t cudaStatus = cudaMalloc((void**)&Md, m * Width * sizeof(float));
    cudaMalloc((void**)&Nd, Width * n * sizeof(float));
    cudaMalloc((void**)&Pd, m * n * sizeof(float));

    std::random_device rd;  // 使用硬件随机数生成器来获得种子
    std::mt19937 gen(rd()); // 使用 Mersenne Twister 引擎
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    // 初始化矩阵内容（示例中简化为全 1 矩阵）
    for (int i = 0; i < m * Width; ++i) {
        hostMd[i] = dist(gen);
    }
    for (int i = 0; i < Width * n; ++i) {
        hostNd[i] = dist(gen);
    }
    // MatrixMulHost(hostMd, hostNd, hostAns, m, n, Width);
    cudaMemcpy(Md, hostMd, m * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Nd, hostNd, n * Width * sizeof(float), cudaMemcpyHostToDevice);

    // 启动核函数
    dim3 threadsPerBlock(16, 16); // 每个线程块包含16x16个线程
    dim3 numBlocks(m/16, n/16); // 根据矩阵大小设置线程块数量

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // 记录开始时间
    MatrixMulKernel<<<numBlocks, threadsPerBlock>>>(Md, Nd, Pd, m, n, Width);
    cudaEventRecord(stop); // 记录结束时间
    cudaEventSynchronize(stop);
    cudaMemcpy(hostAns, Pd, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution MatrixMulKernel Time: %f ms\n", milliseconds);
    
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1); // 记录开始时间
    // 启动核函数
    const int block_size_x_v2 = 16;
    const int block_size_y_v2 = 16;
    dim3 threadsPerBlockV2(block_size_x_v2, block_size_y_v2); // 每个线程块包含16x16个线程
    dim3 numBlocksV2(m / block_size_x_v2, n / block_size_y_v2); // 根据矩阵大小设置线程块数量
    MatrixMulV2Kernel<<<numBlocksV2, threadsPerBlockV2>>>(Md, Nd, Pd, m, n, Width);
    cudaEventRecord(stop1); // 记录结束时间
    cudaEventSynchronize(stop1);
    float milliseconds_v2 = 0;
    cudaEventElapsedTime(&milliseconds_v2, start1, stop1);
    printf("Execution MatrixMulV2Kernel Time: %f ms\n", milliseconds_v2);


    testMatrixMulV3Kernel(Md,Nd,Pd,m,n,Width);
    cudaMemcpy(hostPd, Pd, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // MatrixMulHost(hostMd,hostNd,hostAns,m,n,Width);
    // testCublas(Md,Nd,Pd,m,n,Width);
    for (int i = 0; i < m; ++i) {
        bool flag = true;
        for (int j = 0; j < n; ++j) {
            if (fabs(hostAns[i * n + j] - hostPd[i * n + j]) > 1e-3) {
                printf("%d %d : %f not equal %f\n", i,j,hostAns[i * n + j], hostPd[i * n + j]);
                flag = false;
                break;
            }
            // printf("%f\n",hostAns[i * n + j]);
        }
        if (!flag) break;
    }

    // 释放内存和 CUDA 事件
    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);
    free(hostMd);
    free(hostNd);
    free(hostPd);
    free(hostAns);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
