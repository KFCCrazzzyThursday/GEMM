#include "../include/kernels.cuh"

__global__ void conflictFreeGEMM_sn8(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    const int M, const int N, const int K)
{

    // BLOCK = sm/ty * sk/tx
    const int sm_ty = BLOCK_SIZE;
    const int sk_tx = BLOCK_SIZE;
    // Tunable sn.
    const int sn = 8;
    // Per-thread tile size.
    const int tx = 8;
    const int ty = 8;
    // sm = BLOCK_SIZE * ty, sk = BLOCK_SIZE * tx
    const int sm = sm_ty * ty;
    const int sk = sk_tx * tx;
    // Block indices.
    // const int M_ = M / sm;
    const int N_ = N / sn;
    // const int K_ = K / sk;
    const int B_x = blockIdx.x;
    const int B_y = blockIdx.y;
    const int T_x = threadIdx.x;
    const int T_y = threadIdx.y;
    // Thread index in the block.
    const int thread_index = T_y * blockDim.x + T_x;
    // SMEM
    // To reduce bank conflicts, store A by column and B by row in shared memory.
    __shared__ float A_smem[sn][sm];
    __shared__ float B_smem[sn][sk];

    // Register foreach thread
    const int read_per_thread = sn * ty / BLOCK_SIZE;
    // Register staging for global-to-shared loads.
    float A_reg_load[read_per_thread];
    float B_reg_load[read_per_thread];
    // Registers used in the compute stage.
    float A_reg_compute[ty];
    float B_reg_compute[tx];
    float C_reg[ty][tx] = {0.0f};

    // Shared-memory load indices.

    const int a_smem_y = thread_index * ty / BLOCK_SIZE;
    const int a_smem_x = (thread_index * read_per_thread) % sn;
    const int b_smem_y = thread_index * read_per_thread / sk;
    const int b_smem_x = (thread_index * read_per_thread) % sk;
    const int a_gmem_y = B_y * sm + a_smem_y;
    const int b_gmem_x = B_x * sk + b_smem_x;

    for (int n = 0; n < N_; ++n)
    {
        int a_gmem_x = n * sn + a_smem_x;
        int b_gmem_y = n * sn + b_smem_y;
        int a_gmem_addr = OFFSET(a_gmem_y, a_gmem_x, N);
        int b_gmem_addr = OFFSET(b_gmem_y, b_gmem_x, K);

        // Use register staging to hide global-memory latency.
#pragma unroll
        for (int index = 0; index < read_per_thread; index += 4)
        {
            FLOAT4(A_reg_load[index]) = FLOAT4(a[a_gmem_addr + index]);
            FLOAT4(B_reg_load[index]) = FLOAT4(b[b_gmem_addr + index]);
        }
        // Move staged values from registers to shared memory.
#pragma unroll
        for (int index = 0; index < read_per_thread; index += 4)
        {
            A_smem[a_smem_x + index][a_smem_y] = A_reg_load[index];
            A_smem[a_smem_x + index + 1][a_smem_y] = A_reg_load[index + 1];
            A_smem[a_smem_x + index + 2][a_smem_y] = A_reg_load[index + 2];
            A_smem[a_smem_x + index + 3][a_smem_y] = A_reg_load[index + 3];
            FLOAT4(B_smem[b_smem_y][b_smem_x + index]) = FLOAT4(B_reg_load[index]);
        }
        __syncthreads();

        // Compute C_reg.
#pragma unroll
        for (int tn = 0; tn < sn; tn++)
        {
            // Load one A column and one B row from shared memory into registers.
            // A was transposed when stored, so both loads are row-wise here.
            FLOAT4(A_reg_compute[0]) = FLOAT4(A_smem[tn][T_y * ty / 2]);
            FLOAT4(A_reg_compute[4]) = FLOAT4(A_smem[tn][T_y * ty / 2 + sm / 2]);
            FLOAT4(B_reg_compute[0]) = FLOAT4(B_smem[tn][T_x * tx / 2]);
            FLOAT4(B_reg_compute[4]) = FLOAT4(B_smem[tn][T_x * tx / 2 + sm / 2]);

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
        __syncthreads();
    }

    // Store the upper two result tiles.
#pragma unroll
    for (int ri = 0; ri < ty / 2; ri++)
    {
        int store_c_gmem_m = B_y * sm + T_y * ty / 2 + ri;
        int store_c_gmem_n = B_x * sk + T_x * tx / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, K);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(C_reg[ri][0]);
        FLOAT4(c[store_c_gmem_addr + sk / 2]) = FLOAT4(C_reg[ri][4]);
    }
    // Store the lower two result tiles.
#pragma unroll
    for (int ri = 0; ri < ty / 2; ri++)
    {
        int store_c_gmem_m = B_y * sm + T_y * ty / 2 + ri + sm / 2;
        int store_c_gmem_n = B_x * sk + T_x * tx / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, K);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(C_reg[ri + ty / 2][0]);
        FLOAT4(c[store_c_gmem_addr + sk / 2]) = FLOAT4(C_reg[ri + ty / 2][4]);
    }
}

__global__ void conflictFreeGEMM_sn16(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    const int M, const int N, const int K)
{

    // BLOCK = sm/ty * sk/tx
    const int sm_ty = BLOCK_SIZE;
    const int sk_tx = BLOCK_SIZE;
    // Tunable sn.
    const int sn = 16;
    // Per-thread tile size.
    const int tx = 8;
    const int ty = 8;
    // sm = BLOCK_SIZE * ty, sk = BLOCK_SIZE * tx
    const int sm = sm_ty * ty;
    const int sk = sk_tx * tx;
    // Block indices.
    // const int M_ = M / sm;
    const int N_ = N / sn;
    // const int K_ = K / sk;
    const int B_x = blockIdx.x;
    const int B_y = blockIdx.y;
    const int T_x = threadIdx.x;
    const int T_y = threadIdx.y;
    // Thread index in the block.
    const int thread_index = T_y * blockDim.x + T_x;
    // SMEM
    // To reduce bank conflicts, store A by column and B by row in shared memory.
    __shared__ float A_smem[sn][sm];
    __shared__ float B_smem[sn][sk];

    // Register foreach thread
    const int read_per_thread = sn * ty / BLOCK_SIZE;
    // Register staging for global-to-shared loads.
    float A_reg_load[read_per_thread];
    float B_reg_load[read_per_thread];
    // Registers used in the compute stage.
    float A_reg_compute[ty];
    float B_reg_compute[tx];
    float C_reg[ty][tx] = {0.0f};

    // Shared-memory load indices.

    const int a_smem_y = thread_index * ty / BLOCK_SIZE;
    const int a_smem_x = (thread_index * read_per_thread) % sn;
    const int b_smem_y = thread_index * read_per_thread / sk;
    const int b_smem_x = (thread_index * read_per_thread) % sk;
    const int a_gmem_y = B_y * sm + a_smem_y;
    const int b_gmem_x = B_x * sk + b_smem_x;

    for (int n = 0; n < N_; ++n)
    {
        int a_gmem_x = n * sn + a_smem_x;
        int b_gmem_y = n * sn + b_smem_y;
        int a_gmem_addr = OFFSET(a_gmem_y, a_gmem_x, N);
        int b_gmem_addr = OFFSET(b_gmem_y, b_gmem_x, K);

        // Use register staging to hide global-memory latency.
#pragma unroll
        for (int index = 0; index < read_per_thread; index += 4)
        {
            FLOAT4(A_reg_load[index]) = FLOAT4(a[a_gmem_addr + index]);
            FLOAT4(B_reg_load[index]) = FLOAT4(b[b_gmem_addr + index]);
        }
        // Move staged values from registers to shared memory.
#pragma unroll
        for (int index = 0; index < read_per_thread; index += 4)
        {
            A_smem[a_smem_x + index][a_smem_y] = A_reg_load[index];
            A_smem[a_smem_x + index + 1][a_smem_y] = A_reg_load[index + 1];
            A_smem[a_smem_x + index + 2][a_smem_y] = A_reg_load[index + 2];
            A_smem[a_smem_x + index + 3][a_smem_y] = A_reg_load[index + 3];
            FLOAT4(B_smem[b_smem_y][b_smem_x + index]) = FLOAT4(B_reg_load[index]);
        }
        __syncthreads();

        // Compute C_reg.
#pragma unroll
        for (int tn = 0; tn < sn; tn++)
        {
            // Load one A column and one B row from shared memory into registers.
            // A was transposed when stored, so both loads are row-wise here.
            FLOAT4(A_reg_compute[0]) = FLOAT4(A_smem[tn][T_y * ty / 2]);
            FLOAT4(A_reg_compute[4]) = FLOAT4(A_smem[tn][T_y * ty / 2 + sm / 2]);
            FLOAT4(B_reg_compute[0]) = FLOAT4(B_smem[tn][T_x * tx / 2]);
            FLOAT4(B_reg_compute[4]) = FLOAT4(B_smem[tn][T_x * tx / 2 + sm / 2]);

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

        __syncthreads();
    }

    // Store the upper two result tiles.
#pragma unroll
    for (int ri = 0; ri < ty / 2; ri++)
    {
        int store_c_gmem_m = B_y * sm + T_y * ty / 2 + ri;
        int store_c_gmem_n = B_x * sk + T_x * tx / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, K);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(C_reg[ri][0]);
        FLOAT4(c[store_c_gmem_addr + sk / 2]) = FLOAT4(C_reg[ri][4]);
    }
    // Store the lower two result tiles.
#pragma unroll
    for (int ri = 0; ri < ty / 2; ri++)
    {
        int store_c_gmem_m = B_y * sm + T_y * ty / 2 + ri + sm / 2;
        int store_c_gmem_n = B_x * sk + T_x * tx / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, K);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(C_reg[ri + ty / 2][0]);
        FLOAT4(c[store_c_gmem_addr + sk / 2]) = FLOAT4(C_reg[ri + ty / 2][4]);
    }
}
