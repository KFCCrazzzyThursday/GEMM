#include <cstdio>
#include "../include/kernels.cuh"

__global__ void vectorized_blockGEMM_sn8(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    const int M, const int N, const int K)
{
    // BLOCK = sm/ty * sk/tx
    const int sm_ty = BLOCK_SIZE;
    const int sk_tx = BLOCK_SIZE;
    // 可调
    const int sn = 8;
    // 每个线程负责计算的子矩阵大小tx*ty
    const int tx = 8;
    const int ty = 8;
    // sm = BLOCK_SIZE * ty, sk = BLOCK_SIZE * tx
    const int sm = sm_ty * ty;
    const int sk = sk_tx * tx;
    // 计算BLOCK的索引
    // const int M_ = M / sm;
    const int N_ = N / sn;
    // const int K_ = K / sk;
    const int B_x = blockIdx.x;
    const int B_y = blockIdx.y;
    const int T_x = threadIdx.x;
    const int T_y = threadIdx.y;
    // 计算线程的索引
    const int thread_index = T_y * blockDim.x + T_x;
    // SMEM
    __shared__ float A_smem[sm][sn];
    __shared__ float B_smem[sn][sk];
    // Register per thread
    float C_reg[ty][tx] = {0.0f};

    // 计算从GMEM中加载到SMEM的索引
    const int read_per_thread = sn * ty / BLOCK_SIZE;
    const int a_smem_y = thread_index * ty / BLOCK_SIZE;
    const int a_smem_x = (thread_index * read_per_thread) % sn;
    const int b_smem_y = thread_index * read_per_thread / sk;
    const int b_smem_x = (thread_index * read_per_thread) % sk;
    const int a_gmem_y = B_y * sm + a_smem_y;
    const int b_gmem_x = B_x * sk + b_smem_x;

    for (int n = 0; n < N_; ++n)
    {
        // 把数据从GMEM加载到SMEM
        // 计算与n相关的index
        int a_gmem_x = n * sn + a_smem_x;
        int b_gmem_y = n * sn + b_smem_y;
        int a_gmem_addr = OFFSET(a_gmem_y, a_gmem_x, N);
        int b_gmem_addr = OFFSET(b_gmem_y, b_gmem_x, K);
        #pragma unroll
        for (int index = 0; index < read_per_thread; index += 4)
        {
            FLOAT4(A_smem[a_smem_y][a_smem_x + index]) = FLOAT4(a[a_gmem_addr + index]);
            FLOAT4(B_smem[b_smem_y][b_smem_x + index]) = FLOAT4(b[b_gmem_addr + index]);
        }
        __syncthreads();

// 计算C_reg
#pragma unroll
        for (int tn = 0; tn < sn; ++tn)
        {
#pragma unroll
            for (int tm = 0; tm < ty; ++tm)
            {
#pragma unroll
                for (int tk = 0; tk < tx; ++tk)
                {
                    int a_smem_addr = T_y * ty + tm;
                    int b_smem_addr = T_x * tx + tk;
                    C_reg[tm][tk] += A_smem[a_smem_addr][tn] * B_smem[tn][b_smem_addr];
                }
            }
        }
        __syncthreads();
    }

    // 把C_reg写回GMEM
    for (int ri = 0; ri < ty; ++ri)
    {
        int C_gmem_y = B_y * sm + T_y * ty + ri;
#pragma unroll
        for (int rj = 0; rj < tx; rj += 4)
        {
            int C_gmem_x = B_x * sk + T_x * tx + rj;
            int C_gmem_addr = OFFSET(C_gmem_y, C_gmem_x, K);
            FLOAT4(c[C_gmem_addr]) = FLOAT4(C_reg[ri][rj]);
        }
    }
}

__global__ void vectorized_blockGEMM_sn16(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    const int M, const int N, const int K)
{
    // BLOCK = sm/ty * sk/tx
    const int sm_ty = BLOCK_SIZE;
    const int sk_tx = BLOCK_SIZE;
    // 可调
    const int sn = 16;
    // 每个线程负责计算的子矩阵大小tx*ty
    const int tx = 8;
    const int ty = 8;
    // sm = BLOCK_SIZE * ty, sk = BLOCK_SIZE * tx
    const int sm = sm_ty * ty;
    const int sk = sk_tx * tx;
    // 计算BLOCK的索引
    // const int M_ = M / sm;
    const int N_ = N / sn;
    // const int K_ = K / sk;
    const int B_x = blockIdx.x;
    const int B_y = blockIdx.y;
    const int T_x = threadIdx.x;
    const int T_y = threadIdx.y;
    // 计算线程的索引
    const int thread_index = T_y * blockDim.x + T_x;
    // SMEM
    __shared__ float A_smem[sm][sn];
    __shared__ float B_smem[sn][sk];
    // Register per thread
    float C_reg[ty][tx] = {0.0f};

    // 计算从GMEM中加载到SMEM的索引
    const int read_per_thread = sn * ty / BLOCK_SIZE;
    const int a_smem_y = thread_index * ty / BLOCK_SIZE;
    const int a_smem_x = (thread_index * read_per_thread) % sn;
    const int b_smem_y = thread_index * read_per_thread / sk;
    const int b_smem_x = (thread_index * read_per_thread) % sk;
    const int a_gmem_y = B_y * sm + a_smem_y;
    const int b_gmem_x = B_x * sk + b_smem_x;

    for (int n = 0; n < N_; ++n)
    {
        // 把数据从GMEM加载到SMEM
        // 计算与n相关的index
        int a_gmem_x = n * sn + a_smem_x;
        int b_gmem_y = n * sn + b_smem_y;
        int a_gmem_addr = OFFSET(a_gmem_y, a_gmem_x, N);
        int b_gmem_addr = OFFSET(b_gmem_y, b_gmem_x, K);
        #pragma unroll
        for (int index = 0; index < read_per_thread; index += 4)
        {
            FLOAT4(A_smem[a_smem_y][a_smem_x + index]) = FLOAT4(a[a_gmem_addr + index]);
            FLOAT4(B_smem[b_smem_y][b_smem_x + index]) = FLOAT4(b[b_gmem_addr + index]);
            // FLOAT4(A_smem[a_smem_y + (a_smem_x + index) / sn][(a_smem_x + index) % sn]) = FLOAT4(a[a_gmem_addr + index]);
            // FLOAT4(B_smem[b_smem_y +(b_smem_x + index) / sk][(b_smem_x + index) % sk]) = FLOAT4(b[b_gmem_addr + index]);
        }
        __syncthreads();

// 计算C_reg
#pragma unroll
        for (int tn = 0; tn < sn; ++tn)
        {
#pragma unroll
            for (int tm = 0; tm < ty; ++tm)
            {
#pragma unroll
                for (int tk = 0; tk < tx; ++tk)
                {
                    int a_smem_addr = T_y * ty + tm;
                    int b_smem_addr = T_x * tx + tk;
                    C_reg[tm][tk] += A_smem[a_smem_addr][tn] * B_smem[tn][b_smem_addr];
                }
            }
        }
        __syncthreads();
    }

    // 把C_reg写回GMEM
    for (int ri = 0; ri < ty; ++ri)
    {
        int C_gmem_y = B_y * sm + T_y * ty + ri;
#pragma unroll
        for (int rj = 0; rj < tx; rj += 4)
        {
            int C_gmem_x = B_x * sk + T_x * tx + rj;
            int C_gmem_addr = OFFSET(C_gmem_y, C_gmem_x, K);
            FLOAT4(c[C_gmem_addr]) = FLOAT4(C_reg[ri][rj]);
        }
    }
}