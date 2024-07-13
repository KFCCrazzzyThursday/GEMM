#include <pthread.h>
#include <cmath>
#include "Matrix.hpp"
extern Matrix A, B, C;

struct ThreadTask {
    int A_row_start;
    int A_row_end;
    int B_col_start;
    int B_col_end;
};

class ThreadTasks
{
private:
    ThreadTask *thread_tasks;

public:
    ThreadTasks() : thread_tasks(nullptr) {}
    ThreadTasks(long int thread_count)
    {
        thread_tasks = new ThreadTask[thread_count];
        int m = A.get_rows();
        int k = B.get_cols();
        int A_partitions = 0;
        int B_partitions = 0;
        int sqrt_n = static_cast<int>(std::sqrt(thread_count + 0.1)); // 计算平方根并向下取整
        for (int factor = sqrt_n; factor >= 1; --factor) {
            if (thread_count % factor == 0) {
                A_partitions = thread_count / factor;
                B_partitions = factor;
                break;
            }
        }
        int A_rows_per_thread = m / A_partitions;
        int B_cols_per_thread = k / B_partitions;
        //std::cout << "MMM" << m << ' ' << A_partitions << ' ' << A_rows_per_thread << ' ' << (int)m / A_partitions << std::endl;
        //std::cout << "A B PER THREAD " << A_rows_per_thread << ' ' << B_cols_per_thread << std::endl;
        for (long int thread = 0; thread < thread_count; ++thread) {
            int i = thread / B_partitions;
            int j = thread % B_partitions;
            if (i == A_partitions - 1) {
                thread_tasks[thread].A_row_start = i * A_rows_per_thread;
                thread_tasks[thread].A_row_end = m - 1;
            } else {
                thread_tasks[thread].A_row_start = i * A_rows_per_thread;
                thread_tasks[thread].A_row_end = (i + 1) * A_rows_per_thread - 1;
            }

            if (j == B_partitions - 1) {
                thread_tasks[thread].B_col_start = j * B_cols_per_thread;
                thread_tasks[thread].B_col_end = k - 1;
            } else {
                thread_tasks[thread].B_col_start = j * B_cols_per_thread;
                thread_tasks[thread].B_col_end = (j + 1) * B_cols_per_thread - 1;
            }
        }
    }
    ~ThreadTasks()
    {
        delete[] thread_tasks;
        thread_tasks = nullptr;
    }
    ThreadTask &operator[](int index)
    {
        return this->thread_tasks[index];
    }
};

void *pthread_matmul(void *arg);