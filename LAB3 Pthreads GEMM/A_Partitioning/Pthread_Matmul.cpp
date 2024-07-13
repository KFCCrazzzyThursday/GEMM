# include "Pthread_Matmul.hpp"

void *pthread_matmul(void *arg)
{
    ThreadTask *task = (ThreadTask *)arg;

    int row_start = task->row_start;
    int row_end = task->row_end;
    const double *const local_A = &(A.MAT[row_start * A.cols]);
    const double *const local_B = B.MAT;
    double *local_C = &(C.MAT[row_start * C.cols]);
    const int m = row_end - row_start + 1;
    const int n = A.cols;
    const int k = B.cols;

    sequential_matmul(local_A, local_B, local_C, m, n, k);
    return nullptr;
}