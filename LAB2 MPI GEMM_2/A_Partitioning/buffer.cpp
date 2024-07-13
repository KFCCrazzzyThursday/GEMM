#include "Matrix.hpp"
#include "buffer.hpp"

int BUFFER::size_alignment()
{
    int pad_rows = (this->comm_size - this->A_rows_per_process % this->comm_size) %
                   this->comm_size;
    int new_m = this->A_rows_per_process + pad_rows;
    this->A_rows_per_process = new_m / comm_size;
    return pad_rows;
}

void BUFFER::buffer_init(Matrix &A, const Matrix &B,const int pad)
{
    Matrix temp_A(A);
    temp_A.row_padding(pad);
    int A_size_per_process = Get_A_size_per_process();
    int B_size_per_process = Get_B_size_per_process();
    int size_per_process = A_size_per_process + B_size_per_process;
    buffer = new double[size_per_process * comm_size]();
    for (int i = 0; i < comm_size; i++) {
        for (int j = 0; j < A_size_per_process; j++) {
            buffer[j + size_per_process * i] = temp_A.MAT[j + A_size_per_process * i];
        }
        for (int j = 0; j < B_size_per_process; ++j) {
            buffer[(j + A_size_per_process) + size_per_process * i] = B.MAT[j];
        }
    }
}