#include "include/utils.hpp"
#include "include/kernels.cuh"

int main(int argc, char *argv[])
{
    std::vector<std::vector<double>> performances;
    std::vector<std::string> kernels = {"CUBLAS", "naiveGEMM", "blockGEMM_sn8", "blockGEMM_sn16", "vec_GEMM_sn8", "vec_GEMM_sn16", "conflictFreeGEMM_sn8", "conflictFreeGEMM_sn16",
    "doubleBufferGEMM_sn8", "doubleBufferGEMM_sn16"};
    // std::vector<std::string> kernels = {"CUBLAS", "naiveGEMM","vec_GEMM_sn16"};
    std::vector<int> MatSize = {128, 256, 512, 1024, 2048};
    int M, N, K;
    int BM = BLOCK_SIZE;
    int BK = BLOCK_SIZE;
    int runs = 200;
    printGpuInfo();
    for (auto &size : MatSize)
    {
        M = size;
        N = size;
        K = size;
        dim3 blockDim(BK, BM);
        dim3 gridDim((K + BK - 1) / BK, (M + BM - 1) / BM);
        performances.emplace_back(cublasGemm_test(M, N, K, runs));
        //performances.emplace_back(test_CUDA_GEMM(naiveGEMM, gridDim, blockDim, M, N, K, runs));
        dim3 gridDim2(K / (8 * BK), M / (8 * BM));
        performances.emplace_back(test_CUDA_GEMM(blockGEMM_sn8, gridDim2, blockDim, M, N, K, runs));
        performances.emplace_back(test_CUDA_GEMM(blockGEMM_sn16, gridDim2, blockDim, M, N, K, runs));
        performances.emplace_back(test_CUDA_GEMM(vectorized_blockGEMM_sn8, gridDim2, blockDim, M, N, K, runs));
        performances.emplace_back(test_CUDA_GEMM(vectorized_blockGEMM_sn16, gridDim2, blockDim, M, N, K, runs));
        performances.emplace_back(test_CUDA_GEMM(conflictFreeGEMM_sn8, gridDim2, blockDim, M, N, K, runs));
        performances.emplace_back(test_CUDA_GEMM(conflictFreeGEMM_sn16, gridDim2, blockDim, M, N, K, runs));
        performances.emplace_back(test_CUDA_GEMM(doubleBufferGEMM_sn8, gridDim2, blockDim, M, N, K, runs));
         performances.emplace_back(test_CUDA_GEMM(doubleBufferGEMM_sn16, gridDim2, blockDim, M, N, K, runs));
        std::cout << "####################################################################" << std::endl;
        std::cout << "Running tests for GEMM" << std::endl;
        std::cout << "Matrix dims MxNxK: " << M << " x " << N << " x " << K << std::endl;
        std::cout << "Block dims:        " << blockDim.x << " x " << blockDim.y << std::endl;
        std::cout << "--------------------------------------------------------------------" << std::endl;
        std::cout << "Algorithm\t            Time\t        TFLOPS\t                P RATIO\t" << std::endl;

        for (int i = 0; i < performances.size(); i++)
        {
            std::cout << std::left << std::setw(22) << kernels[i];
            for (int j = 0; j < performances[i].size(); j++)
            {
                std::cout << std::left << std::setw(18) << std::fixed << std::setprecision(6) << performances[i][j];
            }
            double percentage = performances[i][1] / performances[0][1] * 100;
            std::cout << std::right << std::setw(8) << std::fixed << std::setprecision(2) << percentage << '%' << std::endl;
        }
        std::cout << "--------------------------------------------------------------------" << std::endl;
        performances.clear();
    }

    std::cout << "####################################################################" << std::endl;
    return 0;
}