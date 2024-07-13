#ifndef P_MATMUL
#define P_MATMUL

#include <pthread.h>
#include "Matrix.hpp"
extern Matrix A, B, C;

struct ThreadTask {
    int row_start;
    int row_end;
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
        int rows_per_thread = m / thread_count;
        for (long int thread = 0; thread < thread_count - 1; ++thread) {
            thread_tasks[thread].row_start = thread * rows_per_thread;
            thread_tasks[thread].row_end = (thread + 1) * rows_per_thread - 1;
        }
        thread_tasks[thread_count - 1].row_start = (thread_count - 1) * rows_per_thread;
        thread_tasks[thread_count - 1].row_end = m - 1;
    }
    ~ThreadTasks()
    {
        delete[] thread_tasks;
        thread_tasks = nullptr;
    }
    ThreadTask& operator[](int index)
    {
        return this->thread_tasks[index];
    }
};

void *pthread_matmul(void *arg);

#endif