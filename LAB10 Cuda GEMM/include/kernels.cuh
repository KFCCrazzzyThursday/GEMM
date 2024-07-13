// kernels.hpp
#ifndef KERNELS_H
#define KERNELS_H

#ifndef FLOAT_TYPE
#define FLOAT_TYPE float
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

#define OFFSET(row_id, col_id, row_size) ((row_id) * (row_size) + (col_id))
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

// 声明核函数
__global__ void naiveGEMM(FLOAT_TYPE *__restrict__ a, FLOAT_TYPE *__restrict__ b, FLOAT_TYPE *__restrict__ c, const int M, const int N, const int K);

__global__ void blockGEMM_sn8(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    const int M, const int N, const int K);
__global__ void blockGEMM_sn16(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    const int M, const int N, const int K);

__global__ void vectorized_blockGEMM_sn8(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    const int M, const int N, const int K);
__global__ void vectorized_blockGEMM_sn16(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    const int M, const int N, const int K);

__global__ void conflictFreeGEMM_sn8(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    const int M, const int N, const int K);
__global__ void conflictFreeGEMM_sn16(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    const int M, const int N, const int K);

__global__ void doubleBufferGEMM_sn8(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    const int M, const int N, const int K);
__global__ void doubleBufferGEMM_sn16(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    const int M, const int N, const int K);
#endif
