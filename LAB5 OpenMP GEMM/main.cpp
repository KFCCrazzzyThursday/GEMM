#include "Matrix.hpp"
#include <chrono>
#include <stdlib.h>

int main(int argc, char *argv[])
{

    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << "<num_threads> <schedule type> <chunk_size> m n k" << std::endl;
        return 1;
    }

    // 读取omp参数
    int thread_count = std::strtol(argv[1], NULL, 10);
    std::string sched(argv[2]);
    int chunk_size = std::stoi(argv[3]);
    // 读取矩阵参数
    int m = std::stoi(argv[4]);
    int n = std::stoi(argv[5]);
    int k = std::stoi(argv[6]);
    // 初始化矩阵
    Matrix A(m, n, true);
    Matrix B(n, k, true);
    Matrix C(m, k, false);
    // 调度类型
    omp_sched_t sched_type;
    // 计时变量
    std::chrono::duration<double, std::milli> elapsed;
    std::chrono::_V2::system_clock::time_point start_time, end_time;

    /**
     * 1.若调度类型为'default', 则'omp_matmul'函数只需传入 ABC 和 线程数量. 调度类型和chunk_size由编译器决定
     * 2.若不是'default', 则根据'sched'值给出sched_type.
     *   然后传入ABC, 线程数量, 调度类型 和 chunk_size.
     * */
    if (sched == "default") {
        // 开始计时
        start_time = std::chrono::high_resolution_clock::now();
        omp_matmul(A, B, C, thread_count);
    } else {
        //
        if (sched == "static") {
            sched_type = omp_sched_static;
        } else if (sched == "dynamic") {
            sched_type = omp_sched_dynamic;
        } else if (sched == "guided") {
            sched_type = omp_sched_guided;
        } else {
            std::cerr << "Invalid schedule type specified. Using auto." << std::endl;
            sched_type = omp_sched_auto;
        }
        start_time = std::chrono::high_resolution_clock::now();
        omp_matmul(A, B, C, thread_count, sched_type, chunk_size);
    }

    // 计算耗时
    end_time = std::chrono::high_resolution_clock::now();
    elapsed = end_time - start_time;

    /*   std::cout
          << "import numpy as np\n"
          << "LHS = [\n"
          << A
          << ",\n"
          << B
          << "]\nC = np.array(\n"
          << C
          << ")\n"
          << "print(np.all(np.abs(C - np.matmul(LHS[0], LHS[1])) < 1e-4))"
          << std::endl; */

    // 打印矩阵和耗时
    std::cout
        << "Matrix A: \n"
        << A
        << "Matrix B: \n"
        << B
        << "Matrix C: \n"
        << C;
    std::cout
        << "Time taken for matrix calculation: " << elapsed.count() / 1000 << " seconds\n";

    return 0;
}