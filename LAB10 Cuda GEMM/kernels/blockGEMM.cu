#include "../include/kernels.cuh"
// #include <cstdio>
__global__ void blockGEMM_sn8(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    const int M, const int N, const int K)
{
    // BLOCK = sm/ty * sk/tx
    constexpr int sm_ty = BLOCK_SIZE;
    constexpr int sk_tx = BLOCK_SIZE;
    // 可调的sn
    constexpr int sn = 8;
    // 每个线程负责计算的子矩阵大小tx*ty
    constexpr int tx = 8;
    constexpr int ty = 8;
    // sm = BLOCK_SIZE * ty, sk = BLOCK_SIZE * tx
    constexpr int sm = sm_ty * ty;
    constexpr int sk = sk_tx * tx;
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
    // 一个BLOCK加载A的sm*sn, 共BLOCK_SIZE*BLOCK_SIZE个线程
    // 每个线程负责加载A的sm*sn/BLOCK_SIZE^2 = sn*ty/BLOCK_SIZE个元素
    constexpr int read_per_thread = sn * ty / BLOCK_SIZE;
    const int a_smem_y = thread_index * ty / BLOCK_SIZE;
    const int a_smem_x = (thread_index * read_per_thread) % sn;
    const int b_smem_y = thread_index * read_per_thread / sk;
    const int b_smem_x = (thread_index * read_per_thread) % sk;
    // 计算从GMEM加载的固定index
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
        for (int i = 0; i < read_per_thread; ++i)
        {
            A_smem[a_smem_y][a_smem_x + i] = a[a_gmem_addr + i];
            B_smem[b_smem_y][b_smem_x + i] = b[b_gmem_addr + i];
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
                    // 每次加载A的一列和B的一行
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
            c[C_gmem_addr] = C_reg[ri][rj];
            c[C_gmem_addr+1] = C_reg[ri][rj+1];
            c[C_gmem_addr+2] = C_reg[ri][rj+2];
            c[C_gmem_addr+3] = C_reg[ri][rj+3];
        }
    }
}

__global__ void blockGEMM_sn16(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    const int M, const int N, const int K)
{
    // BLOCK = sm/ty * sk/tx
    constexpr int sm_ty = BLOCK_SIZE;
    constexpr int sk_tx = BLOCK_SIZE;
    // 可调
    constexpr int sn = 8;
    // 每个线程负责计算的子矩阵大小tx*ty
    constexpr int tx = 8;
    constexpr int ty = 8;
    // sm = BLOCK_SIZE * ty, sk = BLOCK_SIZE * tx
    constexpr int sm = sm_ty * ty;
    constexpr int sk = sk_tx * tx;
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
    constexpr int read_per_thread = sn * ty / BLOCK_SIZE;
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
        for (int i = 0; i < read_per_thread; ++i)
        {
            A_smem[a_smem_y][a_smem_x + i] = a[a_gmem_addr + i];
            B_smem[b_smem_y][b_smem_x + i] = b[b_gmem_addr + i];
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
            c[C_gmem_addr] = C_reg[ri][rj];
            c[C_gmem_addr+1] = C_reg[ri][rj+1];
            c[C_gmem_addr+2] = C_reg[ri][rj+2];
            c[C_gmem_addr+3] = C_reg[ri][rj+3];
        }
    }
}