#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>
#include <random>

class Matrix
{
private:
    double *MAT;
    int rows, cols;

public:
    Matrix() : MAT(nullptr), rows(0), cols(0){};
    Matrix(int r, int c, bool rand_init = false) : rows(r), cols(c)
    {
        MAT = new double[r * c]();
        if (rand_init) {
            uniformly_initialize_matrix();
        }
    }
    ~Matrix()
    {
        delete[] MAT;
        MAT = nullptr;
    }
    // 用于初始化矩阵，使用X~U(0.0, 1.0)的概率分布
    void uniformly_initialize_matrix();

    // 打印矩阵
    void print_matrix() const;

    // 交换坐标顺序的串行矩阵乘法，A_{m*p} B_{p*n} = C_{m*n}(不用mnk是为避免变量名重复)
    friend inline void sequential_matmul(const Matrix &A, const Matrix &B, Matrix &C);

    // MPI点对点矩阵乘法, A_{m*n} B_{n*k} = C_{m*k}
    friend void mpi_matmul_main_process(const Matrix &A, const Matrix &B, Matrix &C, double &elapsed, const int comm_size);
    friend void mpi_matmul_worker_process(const int m, const int n, const int k, const int my_rank, const int comm_size);
};

// C += A * B
inline void sequential_matmul(const Matrix &A, const Matrix &B, Matrix &C)
{
    int m = A.rows;
    int n = A.cols;
    int p = B.cols;
    for (int i = 0; i < m; ++i) {
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < p; ++j) {
                C.MAT[i * p + j] += A.MAT[i * n + k] * B.MAT[k * p + j];
            }
        }
    }
}

#endif