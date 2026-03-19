#include "../include/kernels.cuh"

__global__ void naiveGEMM(FLOAT_TYPE * __restrict__ a, FLOAT_TYPE *__restrict__ b, FLOAT_TYPE *__restrict__ c, const int M, const int N, const int K)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M && k < K)
    {
        float psum = 0.0f;
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            psum += a[OFFSET(m,n,N)] * b[OFFSET(n,k,K)];
        }
        c[OFFSET(m,k,K)] = psum;
    }
}