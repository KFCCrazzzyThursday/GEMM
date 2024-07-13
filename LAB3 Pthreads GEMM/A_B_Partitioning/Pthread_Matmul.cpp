#include "Pthread_Matmul.hpp"
void *pthread_matmul(void *arg)
{
    ThreadTask *task = (ThreadTask *)arg;
    const int m = A.get_rows();
    const int n = A.get_cols();
    const int k = B.get_cols();
    std::cout << "A: " << task->A_row_start << ' ' << task->A_row_end << std::endl;
    std::cout << "B: " << task->B_col_start << ' ' << task->B_col_end << std::endl;
    const int A_row_start = task->A_row_start;
    const int A_row_end = task->A_row_end;
    const int B_col_start = task->B_col_start;
    const int B_col_end = task->B_col_end;
    partitioned_sequential_matmul(A.MAT, B.MAT, C.MAT, A_row_start, A_row_end, B_col_start, B_col_end, m, n, k);
    return nullptr;
}