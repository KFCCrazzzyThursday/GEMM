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
    Matrix(const Matrix &mat) : rows(mat.rows), cols(mat.cols)
    {
        MAT = new double[rows * cols];
        std::copy(mat.MAT, mat.MAT + rows * cols, MAT);
    }
    Matrix &operator=(const Matrix &mat)
    {
        if (this != &mat) {
            double *newMAT = new double[mat.rows * mat.cols];
            std::copy(mat.MAT, mat.MAT + mat.rows * mat.cols, newMAT);
            delete[] MAT;
            MAT = newMAT;
            rows = mat.rows;
            cols = mat.cols;
        }
        return *this;
    }

    Matrix(Matrix &&mat) noexcept
        : MAT(mat.MAT), rows(mat.rows), cols(mat.cols)
    {
        mat.MAT = nullptr;
        mat.rows = 0;
        mat.cols = 0;
    }
    Matrix &operator=(Matrix &&mat) noexcept
    {
        if (this != &mat) {
            delete[] MAT;
            MAT = mat.MAT;
            rows = mat.rows;
            cols = mat.cols;

            mat.MAT = nullptr;
            mat.rows = 0;
            mat.cols = 0;
        }
        return *this;
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
    int get_rows();
    int get_cols();
    // 交换坐标顺序的串行矩阵乘法，A_{m*p} B_{p*n} = C_{m*n}(不用mnk是为避免变量名重复)
    friend inline void sequential_matmul(const Matrix &A, const Matrix &B, Matrix &C);

    friend void *pthread_matmul(void *arg);
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