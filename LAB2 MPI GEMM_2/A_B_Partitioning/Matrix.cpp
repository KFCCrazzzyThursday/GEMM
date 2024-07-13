#include "Matrix.hpp"
#include "buffer.hpp"

// A_{m * n} B_{n * p} = C_ { m * p }
void sequential_matmul(const double *A, const double *B, double *C, const int m, const int n, const int p)
{
    for (int i = 0; i < m; ++i) {
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < p; ++j) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
};

void print_mat(const double *mat, const int rows, const int cols){
    std::cout << '[';
    for (int i = 0; i < rows; ++i) {
        std::cout << '[';
        for (int j = 0; j < cols; ++j) {
            std::cout << mat[i * cols + j];
            if (j != cols - 1) {
                std::cout << ", ";
            }
        }

        if (i != rows - 1) {
            std::cout << "]," << std::endl;
        } else {
            std::cout << "]]" << std::endl;
        }
    }
}
// 用于初始化矩阵，使用X~U(0.0, 1.0)的概率分布
void Matrix::uniformly_initialize_matrix()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            this->MAT[i * cols + j] = dis(gen);
        }
    }
    return;
}

// 打印矩阵
void Matrix::print_matrix() const
{
    print_mat(this->MAT, this->rows,this->cols);
}

void Matrix::row_padding(const int pad_rows)
{
    int new_rows = this->rows + pad_rows;
    double *temp_mat = new double[new_rows * this->cols]();

    // 复制原始矩阵数据到新矩阵
    for (int i = 0; i < this->cols * this->rows; ++i) {
        temp_mat[i] = this->MAT[i];
    }
    delete[] this->MAT;
    this->MAT = temp_mat;
    this->rows = new_rows;
    return;
}

void Matrix::col_padding(const int pad_cols)
{
    int new_cols = this->cols + pad_cols;
    double *temp_mat = new double[this->rows * new_cols]();

    // 计算新下标
    auto new_index = [this, new_cols](int old_row, int old_col) -> int {
        return old_row * new_cols + old_col;
    };

    // 复制原始矩阵数据到新矩阵
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            temp_mat[new_index(i, j)] = this->MAT[i * this->cols + j];
        }
    }

    delete[] this->MAT;
    this->MAT = temp_mat;
    this->cols = new_cols;
    return;
}

void Matrix::cut(const int r, const int c)
{
    double *temp_mat = new double[r * c];
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            temp_mat[i * c + j] = this->MAT[i * this->cols + j];
        }
    }
    delete[] this->MAT;
    this->MAT = temp_mat;
    this->rows = r;
    this->cols = c;
}
