#include "../include/kernels.cuh"

__global__ void doubleBufferGEMM_sn8(
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
    // 注意为了减少bank conflict, 此处A按列存储，B按行存储,
    __shared__ float A_smem[2][sn][sm];
    __shared__ float B_smem[2][sn][sk];

    // Register foreach thread
    const int read_per_thread = sn * ty / BLOCK_SIZE;
    // GEME2SMEM用register中转
    float A_reg_load[read_per_thread];
    float B_reg_load[read_per_thread];
    // 计算用的register
    float A_reg_compute[ty];
    float B_reg_compute[tx];
    float C_reg[ty][tx] = {0.0f};

    // 计算从GMEM中加载到SMEM的索引

    const int a_smem_y = thread_index * ty / BLOCK_SIZE;
    const int a_smem_x = (thread_index * read_per_thread) % sn;
    const int b_smem_y = thread_index * read_per_thread / sk;
    const int b_smem_x = (thread_index * read_per_thread) % sk;
    const int a_gmem_y = B_y * sm + a_smem_y;
    const int b_gmem_x = B_x * sk + b_smem_x;
    {
        int a_gmem_x = a_smem_x;
        int b_gmem_y = b_smem_y;
        int a_gmem_addr = OFFSET(a_gmem_y, a_gmem_x, N);
        int b_gmem_addr = OFFSET(b_gmem_y, b_gmem_x, K);
#pragma unroll
        for (int index = 0; index < read_per_thread; index += 4)
        {
            FLOAT4(A_reg_load[index]) = FLOAT4(a[a_gmem_addr + index]);
            FLOAT4(B_reg_load[index]) = FLOAT4(b[b_gmem_addr + index]);
        }
#pragma unroll
        for (int index = 0; index < read_per_thread; index += 4)
        {
            A_smem[0][a_smem_x + index][a_smem_y] = A_reg_load[index];
            A_smem[0][a_smem_x + index + 1][a_smem_y] = A_reg_load[index + 1];
            A_smem[0][a_smem_x + index + 2][a_smem_y] = A_reg_load[index + 2];
            A_smem[0][a_smem_x + index + 3][a_smem_y] = A_reg_load[index + 3];
            FLOAT4(B_smem[0][b_smem_y][b_smem_x + index]) = FLOAT4(B_reg_load[index]);
        }
        __syncthreads();
    }

    for (int n = 1; n < N_; ++n)
    {

        int pointer = (n - 1) % 2;
        int pointer_next = n % 2;
        int a_gmem_x = n * sn + a_smem_x;
        int b_gmem_y = n * sn + b_smem_y;
        int a_gmem_addr = OFFSET(a_gmem_y, a_gmem_x, N);
        int b_gmem_addr = OFFSET(b_gmem_y, b_gmem_x, K);

        // 使用register中转掩盖GMEM latency
#pragma unroll
        for (int index = 0; index < read_per_thread; index += 4)
        {
            FLOAT4(A_reg_load[index]) = FLOAT4(a[a_gmem_addr + index]);
            FLOAT4(B_reg_load[index]) = FLOAT4(b[b_gmem_addr + index]);
        }
        // 从register中转到SMEM
                // 计算C_reg
#pragma unroll
        for (int tn = 0; tn < sn; tn++)
        {
            // 从SMEM中加载到register
            // 取出A的对应列和B的对应行
            // 由于A在存入时转置了,所以在SMEM中都是按行取出
            FLOAT4(A_reg_compute[0]) = FLOAT4(A_smem[pointer][tn][T_y * ty / 2]);
            FLOAT4(A_reg_compute[4]) = FLOAT4(A_smem[pointer][tn][T_y * ty / 2 + sm / 2]);
            FLOAT4(B_reg_compute[0]) = FLOAT4(B_smem[pointer][tn][T_x * tx / 2]);
            FLOAT4(B_reg_compute[4]) = FLOAT4(B_smem[pointer][tn][T_x * tx / 2 + sm / 2]);

#pragma unroll
            for (int tm = 0; tm < ty; tm++)
            {
#pragma unroll
                for (int tk = 0; tk < tx; tk++)
                {
                    C_reg[tm][tk] += A_reg_compute[tm] * B_reg_compute[tk];
                }
            }
        }
#pragma unroll
        for (int index = 0; index < read_per_thread; index += 4)
        {
            A_smem[pointer_next][a_smem_x + index][a_smem_y] = A_reg_load[index];
            A_smem[pointer_next][a_smem_x + index + 1][a_smem_y] = A_reg_load[index + 1];
            A_smem[pointer_next][a_smem_x + index + 2][a_smem_y] = A_reg_load[index + 2];
            A_smem[pointer_next][a_smem_x + index + 3][a_smem_y] = A_reg_load[index + 3];
            FLOAT4(B_smem[pointer_next][b_smem_y][b_smem_x + index]) = FLOAT4(B_reg_load[index]);
        }
        __syncthreads();
    }
#pragma unroll
        for (int tn = 0; tn < sn; tn++)
        {
            // 从SMEM中加载到register
            // 取出A的对应列和B的对应行
            // 由于A在存入时转置了,所以在SMEM中都是按行取出
            FLOAT4(A_reg_compute[0]) = FLOAT4(A_smem[1][tn][T_y * ty / 2]);
            FLOAT4(A_reg_compute[4]) = FLOAT4(A_smem[1][tn][T_y * ty / 2 + sm / 2]);
            FLOAT4(B_reg_compute[0]) = FLOAT4(B_smem[1][tn][T_x * tx / 2]);
            FLOAT4(B_reg_compute[4]) = FLOAT4(B_smem[1][tn][T_x * tx / 2 + sm / 2]);

#pragma unroll
            for (int tm = 0; tm < ty; tm++)
            {
#pragma unroll
                for (int tk = 0; tk < tx; tk++)
                {
                    C_reg[tm][tk] += A_reg_compute[tm] * B_reg_compute[tk];
                }
            }
        }
    // 写回上面两块tiles的计算结果
#pragma unroll
    for (int ri = 0; ri < ty / 2; ri++)
    {
        int store_c_gmem_m = B_y * sm + T_y * ty / 2 + ri;
        int store_c_gmem_n = B_x * sk + T_x * tx / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(C_reg[ri][0]);
        FLOAT4(c[store_c_gmem_addr + sk / 2]) = FLOAT4(C_reg[ri][4]);
    }
    // 写回下面两块tiles的计算结果
#pragma unroll
    for (int ri = 0; ri < ty / 2; ri++)
    {
        int store_c_gmem_m = B_y * sm + T_y * ty / 2 + ri + sm / 2;
        int store_c_gmem_n = B_x * sk + T_x * tx / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(C_reg[ri + ty / 2][0]);
        FLOAT4(c[store_c_gmem_addr + sk / 2]) = FLOAT4(C_reg[ri + ty / 2][4]);
    }
}
__global__ void doubleBufferGEMM_sn16(
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
    // 注意为了减少bank conflict, 此处A按列存储，B按行存储,
    __shared__ float A_smem[2][sn][sm];
    __shared__ float B_smem[2][sn][sk];

    // Register foreach thread
    const int read_per_thread = sn * ty / BLOCK_SIZE;
    // GEME2SMEM用register中转
    float A_reg_load[read_per_thread];
    float B_reg_load[read_per_thread];
    // 计算用的register
    float A_reg_compute[ty];
    float B_reg_compute[tx];
    float C_reg[ty][tx] = {0.0f};

    // 计算从GMEM中加载到SMEM的索引

    const int a_smem_y = thread_index * ty / BLOCK_SIZE;
    const int a_smem_x = (thread_index * read_per_thread) % sn;
    const int b_smem_y = thread_index * read_per_thread / sk;
    const int b_smem_x = (thread_index * read_per_thread) % sk;
    const int a_gmem_y = B_y * sm + a_smem_y;
    const int b_gmem_x = B_x * sk + b_smem_x;
    {
        int a_gmem_x = a_smem_x;
        int b_gmem_y = b_smem_y;
        int a_gmem_addr = OFFSET(a_gmem_y, a_gmem_x, N);
        int b_gmem_addr = OFFSET(b_gmem_y, b_gmem_x, K);
#pragma unroll
        for (int index = 0; index < read_per_thread; index += 4)
        {
            FLOAT4(A_reg_load[index]) = FLOAT4(a[a_gmem_addr + index]);
            FLOAT4(B_reg_load[index]) = FLOAT4(b[b_gmem_addr + index]);
        }
#pragma unroll
        for (int index = 0; index < read_per_thread; index += 4)
        {
            A_smem[0][a_smem_x + index][a_smem_y] = A_reg_load[index];
            A_smem[0][a_smem_x + index + 1][a_smem_y] = A_reg_load[index + 1];
            A_smem[0][a_smem_x + index + 2][a_smem_y] = A_reg_load[index + 2];
            A_smem[0][a_smem_x + index + 3][a_smem_y] = A_reg_load[index + 3];
            FLOAT4(B_smem[0][b_smem_y][b_smem_x + index]) = FLOAT4(B_reg_load[index]);
        }
        __syncthreads();
    }
    for (int n = 1; n < N_; ++n)
    {

        int pointer = (n - 1) % 2;
        int pointer_next = n % 2;
        int a_gmem_x = n * sn + a_smem_x;
        int b_gmem_y = n * sn + b_smem_y;
        int a_gmem_addr = OFFSET(a_gmem_y, a_gmem_x, N);
        int b_gmem_addr = OFFSET(b_gmem_y, b_gmem_x, K);

        // 使用register中转掩盖GMEM latency
#pragma unroll
        for (int index = 0; index < read_per_thread; index += 4)
        {
            FLOAT4(A_reg_load[index]) = FLOAT4(a[a_gmem_addr + index]);
            FLOAT4(B_reg_load[index]) = FLOAT4(b[b_gmem_addr + index]);
        }
        // 从register中转到SMEM
                // 计算C_reg
#pragma unroll
        for (int tn = 0; tn < sn; tn++)
        {
            // 从SMEM中加载到register
            // 取出A的对应列和B的对应行
            // 由于A在存入时转置了,所以在SMEM中都是按行取出
            FLOAT4(A_reg_compute[0]) = FLOAT4(A_smem[pointer][tn][T_y * ty / 2]);
            FLOAT4(A_reg_compute[4]) = FLOAT4(A_smem[pointer][tn][T_y * ty / 2 + sm / 2]);
            FLOAT4(B_reg_compute[0]) = FLOAT4(B_smem[pointer][tn][T_x * tx / 2]);
            FLOAT4(B_reg_compute[4]) = FLOAT4(B_smem[pointer][tn][T_x * tx / 2 + sm / 2]);

#pragma unroll
            for (int tm = 0; tm < ty; tm++)
            {
#pragma unroll
                for (int tk = 0; tk < tx; tk++)
                {
                    C_reg[tm][tk] += A_reg_compute[tm] * B_reg_compute[tk];
                }
            }
        }
#pragma unroll
        for (int index = 0; index < read_per_thread; index += 4)
        {
            A_smem[pointer_next][a_smem_x + index][a_smem_y] = A_reg_load[index];
            A_smem[pointer_next][a_smem_x + index + 1][a_smem_y] = A_reg_load[index + 1];
            A_smem[pointer_next][a_smem_x + index + 2][a_smem_y] = A_reg_load[index + 2];
            A_smem[pointer_next][a_smem_x + index + 3][a_smem_y] = A_reg_load[index + 3];
            FLOAT4(B_smem[pointer_next][b_smem_y][b_smem_x + index]) = FLOAT4(B_reg_load[index]);
        }
        __syncthreads();
    }
#pragma unroll
        for (int tn = 0; tn < sn; tn++)
        {
            // 从SMEM中加载到register
            // 取出A的对应列和B的对应行
            // 由于A在存入时转置了,所以在SMEM中都是按行取出
            FLOAT4(A_reg_compute[0]) = FLOAT4(A_smem[1][tn][T_y * ty / 2]);
            FLOAT4(A_reg_compute[4]) = FLOAT4(A_smem[1][tn][T_y * ty / 2 + sm / 2]);
            FLOAT4(B_reg_compute[0]) = FLOAT4(B_smem[1][tn][T_x * tx / 2]);
            FLOAT4(B_reg_compute[4]) = FLOAT4(B_smem[1][tn][T_x * tx / 2 + sm / 2]);

#pragma unroll
            for (int tm = 0; tm < ty; tm++)
            {
#pragma unroll
                for (int tk = 0; tk < tx; tk++)
                {
                    C_reg[tm][tk] += A_reg_compute[tm] * B_reg_compute[tk];
                }
            }
        }
    // 写回上面两块tiles的计算结果
#pragma unroll
    for (int ri = 0; ri < ty / 2; ri++)
    {
        int store_c_gmem_m = B_y * sm + T_y * ty / 2 + ri;
        int store_c_gmem_n = B_x * sk + T_x * tx / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(C_reg[ri][0]);
        FLOAT4(c[store_c_gmem_addr + sk / 2]) = FLOAT4(C_reg[ri][4]);
    }
    // 写回下面两块tiles的计算结果
#pragma unroll
    for (int ri = 0; ri < ty / 2; ri++)
    {
        int store_c_gmem_m = B_y * sm + T_y * ty / 2 + ri + sm / 2;
        int store_c_gmem_n = B_x * sk + T_x * tx / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(C_reg[ri + ty / 2][0]);
        FLOAT4(c[store_c_gmem_addr + sk / 2]) = FLOAT4(C_reg[ri + ty / 2][4]);
    }
}