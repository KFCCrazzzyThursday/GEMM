#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <cassert>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>

#include <cuda_runtime.h>
#include <curand_kernel.h>


#include <cublas_v2.h>

#include "kernels.cuh"

// Function to read M, N, K from command line arguments
void readDim(int argc, char *argv[], int &M, int &N, int &K);

// Function to initialize matrices A, B, and C as 1D arrays
void initMat(int M, int N, int K, FLOAT_TYPE *&A, FLOAT_TYPE *&B, FLOAT_TYPE *&C);
void cloneMat(int M, int N, int K, FLOAT_TYPE *srcA, FLOAT_TYPE *srcB, FLOAT_TYPE *&A, FLOAT_TYPE *&B, FLOAT_TYPE *&C);

void printMat(const FLOAT_TYPE *mat, int rows, int cols);


void cublasGemm_f(cublasHandle_t handle, float *A, float *B, float *C, int M, int N, int K);
void cublasGemmTensorCore_f(cublasHandle_t handle, float *A, float *B, float *C, int M, int N, int K);
std::vector<double> test_CUDA_GEMM(
    void (*gemm)(FLOAT_TYPE *, FLOAT_TYPE *, FLOAT_TYPE *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int runs);
std::vector<double> test_CUDA_GEMM(
    void (*gemm)(FLOAT_TYPE *, FLOAT_TYPE *, FLOAT_TYPE *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int runs,
    FLOAT_TYPE *A_ref, FLOAT_TYPE *B_ref);
std::vector<double> cublasGemm_test(const int M, const int N, const int K, const int runs);
std::vector<double> cublasGemmTensorCore_test(const int M, const int N, const int K, const int runs);
std::vector<double> cublasGemm_test(const int M, const int N, const int K, const int runs, FLOAT_TYPE *A_ref, FLOAT_TYPE *B_ref);
std::vector<double> cublasGemmTensorCore_test(const int M, const int N, const int K, const int runs, FLOAT_TYPE *A_ref, FLOAT_TYPE *B_ref);
std::vector<double> benchmarkGpuOp(const std::function<void()> &op, long long flop_count, int warmup, int runs, int repeats);

void printGpuInfo();
#endif // UTILS_HPP
