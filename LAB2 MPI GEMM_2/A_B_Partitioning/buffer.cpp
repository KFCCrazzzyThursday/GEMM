#include "Matrix.hpp"
#include "buffer.hpp"
#include <cmath>

int BUFFER::m_alignment()
{
    //std::cout << "ABLCKS" << A_blocks << std::endl;
    int A_pad_rows = (this->A_blocks - this->A_rows_per_process % this->A_blocks) %
                     this->A_blocks;
    int new_m = this->A_rows_per_process + A_pad_rows;
    this->A_rows_per_process = new_m / A_blocks;
    return A_pad_rows;
}

int BUFFER::k_alignment()
{
    //std::cout<<"BBLCKS"<<B_blocks<<std::endl;
    int B_pad_cols = (this->B_blocks - this->B_cols_per_process % this->B_blocks) %
                     this->B_blocks;
    int new_k = this->B_cols_per_process + B_pad_cols;
    this->B_cols_per_process = new_k / B_blocks;
    return B_pad_cols;
}

void BUFFER::partition()
{
    int sqrt_n = static_cast<int>(std::sqrt(comm_size + 0.1)); // 计算平方根并向下取整
    for (int factor = sqrt_n; factor >= 1; --factor) {
        if (comm_size % factor == 0) {
            this->A_blocks = comm_size / factor;
            this->B_blocks = factor;
            break;
        }
    }
    return;
}

void BUFFER::buffer_init(Matrix &A, const Matrix &B, const int A_pad_rows, const int B_pad_cols)
{
    Matrix temp_A(A);
    Matrix temp_B(B);
    temp_A.row_padding(A_pad_rows);
    temp_B.col_padding(B_pad_cols);
    int A_size_per_process = Get_A_size_per_process();
    int B_size_per_process = Get_B_size_per_process();
    int size_per_process = A_size_per_process + B_size_per_process;
    //std::cout<<"INIT CHECK"<<B_cols_per_process<<' '<<temp_B.cols<<std::endl;
    buffer = new double[size_per_process * comm_size]();
    for (int i = 0; i < A_blocks; ++i) {
        for (int j = 0; j < B_blocks; ++j) {
            for (int k = 0; k < A_size_per_process; ++k) {
                buffer[(i * B_blocks + j) * size_per_process + k] = temp_A.MAT[i * A_size_per_process + k];
            }
            for (int k = A_size_per_process; k < size_per_process; ++k) {
                //std::cout << "k check:" << i << ' ' << j << ' ' << (k - A_size_per_process) << ' ' << (k - A_size_per_process) / B_cols_per_process * temp_B.cols + j * B_cols_per_process + (k - A_size_per_process) % B_cols_per_process << std::endl;
                buffer[(i * B_blocks + j) * size_per_process + k] = temp_B.MAT[(k - A_size_per_process) / B_cols_per_process  * temp_B.cols + j * B_cols_per_process +
                                                                          (k - A_size_per_process)%B_cols_per_process];
            }
        }
    }
    //std::cout<<"a b:"<<A_blocks<<' '<<B_blocks<<' '<<A_size_per_process<<' '<<size_per_process<<std::endl;
    //print_mat(buffer,comm_size,size_per_process);
}