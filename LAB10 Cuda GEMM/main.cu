#include "include/utils.hpp"
#include "include/kernels.cuh"

int main(int argc, char *argv[])
{
    int M, N, K;
    readDim(argc, argv, M, N, K);
    float *A, *B, *C;
    initMat(M, N, K, A, B, C);
    int BM = BLOCK_SIZE;
    int BK = BLOCK_SIZE;
    dim3 blockDim(BK, BM);

    // 计时开始
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // naiveGEMM
    cudaEventRecord(start);
    dim3 gridDim((K + BK - 1) / BK, (M + BM - 1) / BM);
    naiveGEMM<<<gridDim, blockDim>>>(A, B, C, M, N, K);

    // 以下分块乘法算法的gridDim需要改动
    //  dim3 gridDim2(K / (8 * BK), M / (8 * BM));

    // blockGEMM_sn8<<<gridDim2, blockDim>>>(A, B, C, M, N, K);
    // blockGEMM_sn16<<<gridDim2, blockDim>>>(A, B, C, M, N, K);
    // vectorized_blockGEMM_sn8<<<gridDim2, blockDim>>>(A, B, C, M, N, K);
    // vectorized_blockGEMM_sn16<<<gridDim2, blockDim>>>(A, B, C, M, N, K);
    // conflictFreeGEMM_sn8<<<gridDim2, blockDim>>>(A, B, C, M, N, K);
    // conflictFreeGEMM_sn16<<<gridDim2, blockDim>>>(A, B, C, M, N, K);
    // doubleBufferGEMM_sn8<<<gridDim2, blockDim>>>(A, B, C, M, N, K);
    // doubleBufferGEMM_sn16<<<gridDim2, blockDim>>>(A, B, C, M, N, K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GEMM execution time: " << milliseconds << " ms" << std::endl;
    printMat(C, M, K);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}