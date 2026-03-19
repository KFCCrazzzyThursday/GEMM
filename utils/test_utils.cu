#include "../include/utils.hpp"
#include <algorithm>

void cublasGemm_f(cublasHandle_t handle, float *A, float *B, float *C, int M, int N, int K)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, &alpha, B, K, A, N, &beta, C, K);
}

void cublasGemmTensorCore_f(cublasHandle_t handle, float *A, float *B, float *C, int M, int N, int K)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasGemmEx(handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 K, M, N,
                 &alpha,
                 B, CUDA_R_32F, K,
                 A, CUDA_R_32F, N,
                 &beta,
                 C, CUDA_R_32F, K,
                 CUBLAS_COMPUTE_32F_FAST_TF32,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

std::vector<double> cublasGemm_test(const int M, const int N, const int K, const int runs)
{
    FLOAT_TYPE *A_ref, *B_ref, *C_ref;
    initMat(M, N, K, A_ref, B_ref, C_ref);
    auto result = cublasGemm_test(M, N, K, runs, A_ref, B_ref);
    cudaFree(A_ref);
    cudaFree(B_ref);
    cudaFree(C_ref);
    return result;
}

std::vector<double> cublasGemmTensorCore_test(const int M, const int N, const int K, const int runs)
{
    FLOAT_TYPE *A_ref, *B_ref, *C_ref;
    initMat(M, N, K, A_ref, B_ref, C_ref);
    auto result = cublasGemmTensorCore_test(M, N, K, runs, A_ref, B_ref);
    cudaFree(A_ref);
    cudaFree(B_ref);
    cudaFree(C_ref);
    return result;
}

std::vector<double> benchmarkGpuOp(const std::function<void()> &op, long long flop_count, int warmup, int runs, int repeats)
{
    std::vector<double> samples;
    samples.reserve(repeats);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    for (int rep = 0; rep < repeats; ++rep)
    {
        for (int i = 0; i < warmup; ++i)
        {
            op();
        }
        cudaEventRecord(start);
        for (int i = 0; i < runs; ++i)
        {
            op();
        }
        cudaEventRecord(end);
        cudaEventSynchronize(end);

        float msec = 0.0f;
        cudaEventElapsedTime(&msec, start, end);
        samples.push_back(msec / runs);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    std::sort(samples.begin(), samples.end());
    const double median_msec = samples[samples.size() / 2];
    const double tflops = static_cast<double>(flop_count) / (median_msec / 1000.0) / 1e12;
    return {median_msec / 1000.0, tflops};
}

std::vector<double> cublasGemm_test(const int M, const int N, const int K, const int runs, FLOAT_TYPE *A_ref, FLOAT_TYPE *B_ref)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    FLOAT_TYPE *A, *B, *C;
    cloneMat(M, N, K, A_ref, B_ref, A, B, C);

    const int warmup = 50;
    const int repeats = 7;
    auto result = benchmarkGpuOp(
        [&]()
        {
            cublasGemm_f(handle, A, B, C, M, N, K);
        },
        2ll * M * N * K,
        warmup, runs, repeats);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cublasDestroy(handle);

    return result;
}

std::vector<double> cublasGemmTensorCore_test(const int M, const int N, const int K, const int runs, FLOAT_TYPE *A_ref, FLOAT_TYPE *B_ref)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    FLOAT_TYPE *A, *B, *C;
    cloneMat(M, N, K, A_ref, B_ref, A, B, C);

    const int warmup = 50;
    const int repeats = 7;
    auto result = benchmarkGpuOp(
        [&]()
        {
            cublasGemmTensorCore_f(handle, A, B, C, M, N, K);
        },
        2ll * M * N * K,
        warmup, runs, repeats);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cublasDestroy(handle);

    return result;
}

std::vector<double> test_CUDA_GEMM(void (*gemm)(FLOAT_TYPE *, FLOAT_TYPE *, FLOAT_TYPE *, const int, const int, const int), dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int runs)
{
    FLOAT_TYPE *A_ref, *B_ref, *C_ref;
    initMat(M, N, K, A_ref, B_ref, C_ref);
    auto result = test_CUDA_GEMM(gemm, gridDim, blockDim, M, N, K, runs, A_ref, B_ref);
    cudaFree(A_ref);
    cudaFree(B_ref);
    cudaFree(C_ref);
    return result;
}

std::vector<double> test_CUDA_GEMM(void (*gemm)(FLOAT_TYPE *, FLOAT_TYPE *, FLOAT_TYPE *, const int, const int, const int), dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int runs, FLOAT_TYPE *A_ref, FLOAT_TYPE *B_ref)
{
    FLOAT_TYPE *A, *B, *C, *C_cublas;
    cloneMat(M, N, K, A_ref, B_ref, A, B, C);
    cudaMalloc(&C_cublas, M * K * sizeof(FLOAT_TYPE));
    cudaMemset(C_cublas, 0.0, M * K * sizeof(float));
    // Validate correctness.

    // std::cout << "Grid dim: " << gridDim.x << " x " << gridDim.y << std::endl;
    // std::cout << "Block dim: " << blockDim.x << " x " << blockDim.y << std::endl;
    // std::cout << "R" << std::endl;
    // int minGridSize, blockSize;
    // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, mySgemmV1Aligned, 0, 0);
    // std::cout<<"BEST?: "<<minGridSize<<" "<<blockSize<<std::endl;

    // std::cout <<gridDim.x<<" "<<gridDim.y<<" "<<blockDim.x<<" "<<blockDim.y<<std::endl;
    gemm<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        std::cout << "--------------------------------------------------------------------" << std::endl;
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
        cudaFree(C_cublas);
        exit(EXIT_FAILURE);
        return {};
    }
    // std::cout << "E" << std::endl;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasGemm_f(handle, A, B, C_cublas, M, N, K);
    // printMat(C, M, K);
    // Copy results back to host for comparison.
    std::vector<FLOAT_TYPE> hostMat(M * K);
    cudaMemcpy(hostMat.data(), C, M * K * sizeof(FLOAT_TYPE), cudaMemcpyDeviceToHost);
    std::vector<FLOAT_TYPE> hostMat_cublas(M * K);
    cudaMemcpy(hostMat_cublas.data(), C_cublas, M * K * sizeof(FLOAT_TYPE), cudaMemcpyDeviceToHost);
    bool result_is_close = true;
    // double maxError = 0.0;
    // double sumError = 0.0;
    // double maxc;
    // double maxc_cublas;
    // for (int i = 0; i < M * K; ++i) {
    //     double error = fabs(hostMat[i] - hostMat_cublas[i]);
    //     sumError += error;
    //     if (error > maxError) {
    //         maxError = error;
    //         maxc = hostMat[i];
    //         maxc_cublas = hostMat_cublas[i];
    //     }
    // }
    // std::cout <<"MAT E: "<< maxError << "AVG E: " << sumError / (M * K) << std::endl;
    // std::cout << "MAX C: " << maxc << " MAX C CUBLAS: " << maxc_cublas << std::endl;
    for (int i = 0; i < M * K; ++i)
    {
        if (std::fabs(hostMat[i] - hostMat_cublas[i]) > 1e-2)
        {
            result_is_close = false;
            std::cout << "Results do not match at index " << i << ": " << hostMat[i] << " != " << hostMat_cublas[i] << std::endl;
            std::cout << hostMat[i] << " " << hostMat_cublas[i] << std::endl;
            //  printMat(A, M, N);
            //  printMat(B, N, K);
            //  printMat(C, M, K);
            //  printMat(C_cublas, M, K);
            break;
        }
    }

    // printMat(C_cublas, M, K);
    // assert(result_is_close && "do not match!!!!!");
    if(!result_is_close){
        std::cout << "FAILED!" << std::endl;
    }else{
        std::cout << "PASSED!" << std::endl;
    }

    // Time the custom GEMM kernel.
    const int warmup = 50;
    const int repeats = 7;
    auto result = benchmarkGpuOp(
        [&]()
        {
            gemm<<<gridDim, blockDim>>>(A, B, C, M, N, K);
        },
        2ll * M * N * K,
        warmup, runs, repeats);

    // Release memory.
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(C_cublas);
    // Destroy CUDA events.
    cublasDestroy(handle);
    return result;
}

void printGpuInfo()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << ": " << deviceProp.name << std::endl;
        std::cout << "  Total number of SMs:                         " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Maximum number of threads per SM:            " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Maximum number of threads per block:         " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Maximum size of each dimension of a block:   "
                  << deviceProp.maxThreadsDim[0] << " x "
                  << deviceProp.maxThreadsDim[1] << " x "
                  << deviceProp.maxThreadsDim[2] << std::endl;
        std::cout << "  Maximum size of each dimension of a grid:    "
                  << deviceProp.maxGridSize[0] << " x "
                  << deviceProp.maxGridSize[1] << " x "
                  << deviceProp.maxGridSize[2] << std::endl;
        std::cout << "  Shared memory per block:                     " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
        std::cout << "  Total global memory:                         " << deviceProp.totalGlobalMem / pow(2, 30) << " GB" << std::endl;
        std::cout << "  Number of registers per SM:                  " << deviceProp.regsPerMultiprocessor << std::endl;
        std::cout << "  Number of registers per block:               " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  Maximum registers per thread:                "
                  << deviceProp.regsPerBlock / deviceProp.maxThreadsPerBlock << std::endl;
    }
}
