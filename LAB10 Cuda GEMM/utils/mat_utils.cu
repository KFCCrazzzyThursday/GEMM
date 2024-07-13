#include "../include/utils.hpp"


__global__ void initRandomKernel(FLOAT_TYPE *matrix, int size, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        curandState state;
        curand_init(seed, idx, 0, &state);
        matrix[idx] = curand_uniform(&state);
    }
}

__global__ void initZeroKernel(FLOAT_TYPE *matrix, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        matrix[idx] = 0.0;
    }
}

void initMat(int M, int N, int K, FLOAT_TYPE *&A, FLOAT_TYPE *&B, FLOAT_TYPE *&C)
{
    size_t sizeA = M * N * sizeof(FLOAT_TYPE);
    size_t sizeB = N * K * sizeof(FLOAT_TYPE);
    size_t sizeC = M * K * sizeof(FLOAT_TYPE);

    cudaMalloc(&A, sizeA);
    cudaMalloc(&B, sizeB);
    cudaMalloc(&C, sizeC);

    int blockSize = BLOCK_SIZE;
    int numBlocksA = (M * N + blockSize - 1) / blockSize;
    int numBlocksB = (N * K + blockSize - 1) / blockSize;
    int numBlocksC = (M * K + blockSize - 1) / blockSize;

    unsigned long long seed = time(NULL);
    initRandomKernel<<<numBlocksA, blockSize>>>(A, M * N, seed);
    initRandomKernel<<<numBlocksB, blockSize>>>(B, N * K, seed - 1);
    initZeroKernel<<<numBlocksC, blockSize>>>(C, M * K);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void readDim(int argc, char *argv[], int &M, int &N, int &K)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " M N K" << std::endl;
        exit(EXIT_FAILURE);
    }
    M = std::atoi(argv[1]);
    N = std::atoi(argv[2]);
    K = std::atoi(argv[3]);
}

void printMat(const FLOAT_TYPE *mat, int rows, int cols)
{
    std::vector<FLOAT_TYPE> hostMat(rows * cols);
    cudaMemcpy(hostMat.data(), mat, rows * cols * sizeof(FLOAT_TYPE), cudaMemcpyDeviceToHost);

    std::cout << "[";
    for (int i = 0; i < rows; i++)
    {
        if (i > 0)
            std::cout << " ";
        std::cout << "[";
        for (int j = 0; j < cols; j++)
        {
            std::cout << hostMat[i * cols + j];
            if (j < cols - 1)
                std::cout << ", ";
        }
        std::cout << "]";
        if (i < rows - 1)
            std::cout << "," << std::endl;
    }
    std::cout << "]" << std::endl;
}