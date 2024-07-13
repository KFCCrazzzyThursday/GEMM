#include "../include/utils.hpp"

void cublasGemm_f(cublasHandle_t handle, float *A, float *B, float *C, int M, int N, int K)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, &alpha, B, K, A, N, &beta, C, K);
}

std::vector<double> cublasGemm_test(const int M, const int N, const int K, const int runs)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    FLOAT_TYPE *A, *B, *C, *C_cublas;
    initMat(M, N, K, A, B, C);
    cudaMalloc(&C_cublas, M * K * sizeof(FLOAT_TYPE));
    // 初始化 CUDA 事件
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // 记录开始时间
    cudaEventRecord(start);
    for (int i = 0; i < runs; ++i)
    {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, &alpha, B, K, A, N, &beta, C, K);
    }
    // 记录结束时间
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    // 计算执行时间
    float msec;
    cudaEventElapsedTime(&msec, start, end);

    // 计算并打印 TFLOPS
    double TFLOPS = (2.0 * M * N * K * runs) / (msec / 1000.0) / 1e12;

    // 销毁 CUDA 事件
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cublasDestroy(handle);

    return {msec / 1000.0 / runs, TFLOPS};
}

std::vector<double> test_CUDA_GEMM(void (*gemm)(FLOAT_TYPE *, FLOAT_TYPE *, FLOAT_TYPE *, const int, const int, const int), dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int runs)
{
    FLOAT_TYPE *A, *B, *C, *C_cublas;
    initMat(M, N, K, A, B, C);
    cudaMalloc(&C_cublas, M * K * sizeof(FLOAT_TYPE));
    cudaMemset(C_cublas, 0.0, M * K * sizeof(float));
    // 验证正确性

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
    //  复制到CPU上比较
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

    // 执行自定义的GEMM核函数并计时
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < runs; i++)
    {
        gemm<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    // 计算执行时间
    float msec;
    cudaEventElapsedTime(&msec, start, end);

    // 计算并打印 TFLOPS
    // std::cout << msec / 1000.0 / runs << "s\t";
    double TFLOPS = (2.0 * M * N * K * runs) / (msec / 1000.0) / 1e12;

    // 释放内存
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(C_cublas);
    // 销毁 CUDA 事件
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cublasDestroy(handle);
    return {msec / 1000.0 / runs, TFLOPS};
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