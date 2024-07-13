#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>
#include <random>
#include <mpi.h>
class BUFFER;
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
    Matrix(const Matrix &mat) : rows(mat.rows), cols(mat.cols)
    {
        MAT = new double[rows * cols];
        std::copy(mat.MAT, mat.MAT + rows * cols, MAT);
    }

    Matrix(Matrix &&mat) noexcept : MAT(nullptr), rows(0), cols(0)
    {
        MAT = mat.MAT;
        rows = mat.rows;
        cols = mat.cols;

        mat.MAT = nullptr;
        mat.rows = 0;
        mat.cols = 0;
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

    // 矩阵填充, 用于把行/列数量补充成线程数的倍数
    void row_padding(const int pad_rows);
    void col_padding(const int pad_cols);
    void cut(const int r, const int l);
    // 交换坐标顺序的串行矩阵乘法，A_{m*p} B_{p*n} = C_{m*n}(不用mnk是为避免变量名重复)
    friend inline void sequential_matmul(const Matrix &A, const Matrix &B, Matrix &C);

    // MPI集合矩阵乘法, A_{m*n} B_{n*k} = C_{m*k}
    friend void mpi_matmul_main_process(const BUFFER *const buffer, Matrix &C, MPI_Datatype *COMBINED_MATRIX);
    // friend void mpi_matmul_worker_process(const int m, const int n, const int k, const int my_rank, const int comm_size);
    friend class BUFFER;
};

// A_{m * n} B_{n * p} = C_ { m * p }
void sequential_matmul(const double *A, const double *B, double *C, const int m, const int n, const int k);
void print_mat(const double *mat, const int rows, const int cols);
// C += A * B
inline void sequential_matmul(const Matrix &A, const Matrix &B, Matrix &C)
{
    int m = A.rows;
    int n = A.cols;
    int p = B.cols;
    sequential_matmul(A.MAT, B.MAT, C.MAT, m, n, p);
}

#endif