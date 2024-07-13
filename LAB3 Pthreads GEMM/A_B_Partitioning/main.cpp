#include "Pthread_Matmul.hpp"
#include <chrono>
#include <stdlib.h>

long int thread_count;
int m, n, k;
Matrix A, B, C;
int main(int argc, char *argv[])
{

    // 参数数量不正确则在主线程中打印报错并return
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " m n k" << std::endl;
        return 1;
    }

    // 读取矩阵参数
    thread_count = std::strtol(argv[1], NULL, 10);
    m = std::stoi(argv[2]);
    n = std::stoi(argv[3]);
    k = std::stoi(argv[4]);

    // 初始化矩阵
    A = Matrix(m, n, true);
    B = Matrix(n, k, true);
    C = Matrix(m, k, false);
    // 开始计时
    std::chrono::duration<double, std::milli> elapsed;
    auto start_time = std::chrono::high_resolution_clock::now();

    // 线程handles
    pthread_t *thread_handles = new pthread_t[thread_count];

    // 预先分配每个线程的任务
    ThreadTasks thread_tasks(thread_count);

    // 计算矩阵乘法
    for (long thread = 0; thread < thread_count; ++thread) {
        pthread_create(&thread_handles[thread], NULL, pthread_matmul, (void *)&thread_tasks[thread]);
    }
    for (long thread = 0; thread < thread_count; ++thread) {
        pthread_join(thread_handles[thread], NULL);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    //  打印矩阵A B C
    std::cout << "Matrix A:\n";
    A.print_matrix();
    std::cout << "\nMatrix B:\n";
    B.print_matrix();
    std::cout << "\nMatrix C:\n";
    C.print_matrix();
    delete [] thread_handles;
    // 打印耗时
    elapsed = end_time - start_time;
    std::cout << "Time taken for matrix calculation: " << elapsed.count() / 1000 << " seconds\n";
    // 从进程运算

    return 0;
}