#include "Matrix.hpp"

std::ostream &operator<<(std::ostream &os, const Matrix &mat)
{
    os << '[';
    for (int i = 0; i < mat.rows; ++i) {
        if (i != 0) {
            os << ' ';
        }
        os << '[';
        for (int j = 0; j < mat.cols; ++j) {
            os << mat.MAT[i * mat.cols + j];
            if (j != mat.cols - 1) {
                os << ", ";
            }
        }

        if (i != mat.rows - 1) {
            os << "]," << std::endl;
        } else {
            os << "]]" << std::endl;
        }
    }
    return os;
}

/**
 * 使用 OpenMP 并行计算矩阵乘法C_{m, p} += A_{m, n} B_{n, p}，并允许设置调度类型和块大小
 * @param A 指向A
 * @param B 指向B
 * @param C 指向C
 * @param m
 * @param n
 * @param p
 * @param num_threads 线程数
 * @param sched_type 调度类型(omp_sched_static/omp_sched_dynamic/omp_sched_guided/omp_sched_auto)
 * @param chunk_size chunk_size
 *
 * 使用每个线程局部数组 C_local 存储中间结果，以减少对共享内存的竞争使用
 * 最后使用原子操作+=更新全局结果矩阵 C。
 */
void omp_matmul(const double *A, const double *B, double *C,
                int m, int n, int p,
                int num_threads, omp_sched_t sched_type, int chunk_size)
{
    omp_set_num_threads(num_threads);
    omp_set_schedule(sched_type, chunk_size);

    #pragma omp parallel
    {
        std::vector<double> C_local(m * p, 0.0); // 初始化局部数组

        #pragma omp for collapse(2) schedule(runtime)
        for (int i = 0; i < m; ++i) {
            for (int k = 0; k < n; ++k) {
                for (int j = 0; j < p; ++j) {
                    C_local[i * p + j] += A[i * n + k] * B[k * p + j];
                }
            }
        }

        // 使用 atomic 更新全局矩阵 C
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < p; ++j) {
                #pragma omp atomic
                C[i * p + j] += C_local[i * p + j];
            }
        }
    }
}

/**
 * 使用 OpenMP 并行计算矩阵乘法C_{m, p} += A_{m, n} B_{n, p}，使用默认调度
 * @param A 指向A
 * @param B 指向B
 * @param C 指向C
 * @param m
 * @param n
 * @param p
 * @param num_threads 线程数
 *
 * 使用每个线程局部数组 C_local 存储中间结果，以减少对共享内存的竞争使用
 * 最后使用原子操作+=更新全局结果矩阵 C
 * 该函数不允许指定调度策略和块大小，采用编译器默认
 */
void omp_matmul_default(const double *A, const double *B, double *C,
                        int m, int n, int p,
                        int num_threads)
{
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        std::vector<double> C_local(m * p, 0.0); // 初始化局部数组

        #pragma omp for collapse(2)
        for (int i = 0; i < m; ++i) {
            for (int k = 0; k < n; ++k) {
                for (int j = 0; j < p; ++j) {
                    C_local[i * p + j] += A[i * n + k] * B[k * p + j];
                }
            }
        }

        // 使用 atomic 更新全局矩阵 C
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < p; ++j) {
                #pragma omp atomic
                C[i * p + j] += C_local[i * p + j];
            }
        }
    }
}

void print_mat(const double *mat, const int rows, const int cols)
{
    std::cout << '[';
    for (int i = 0; i < rows; ++i) {
        if (i != 0) {
            std::cout << ' ';
        }
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
    print_mat(this->MAT, this->rows, this->cols);
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

int Matrix::get_rows()
{
    return this->rows;
}

int Matrix::get_cols()
{
    return this->cols;
}
