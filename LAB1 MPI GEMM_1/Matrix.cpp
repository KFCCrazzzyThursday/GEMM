#include "Matrix.hpp"
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
    std::cout << '[';
    for (int i = 0; i < rows; ++i) {
        std::cout << '[';
        for (int j = 0; j < cols; ++j) {
            std::cout << this->MAT[i * cols + j];
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