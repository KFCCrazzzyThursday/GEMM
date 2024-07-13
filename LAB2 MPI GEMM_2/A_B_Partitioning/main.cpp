#include "MPI_Collective_Matmul.hpp"
int main(int argc, char *argv[])
{
    // MPI环境初始化 及 进程参数获取
    MPI_Init(&argc, &argv);
    int my_rank = 0;
    int comm_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    // 参数数量不正确则在主进程中打印报错并return
    if (my_rank == 0) {
        if (argc != 4) {
            std::cerr << "Usage: " << argv[0] << " m n k" << std::endl;
            return 1;
        }
    }
    // 读取矩阵参数
    int m = std::stoi(argv[1]);
    int n = std::stoi(argv[2]);
    int k = std::stoi(argv[3]);

    BUFFER Buffer(m, n, k, comm_size);
    // MPI聚合矩阵类型
    MPI_Datatype COMBINED_MATRIX;
    Buffer.partition();
    int A_pad_rows = Buffer.m_alignment();
    int B_pad_cols = Buffer.k_alignment();
    build_mpi_mat_combined_type(&Buffer, &COMBINED_MATRIX);
    // 若为主进程
    if (my_rank == 0) {
        // 初始化矩阵
        Matrix A(m, n, true);
        Matrix B(n, k, true);
        Matrix C(m, k, false);
        // 变量elapsed用于计时
        double start_time = MPI_Wtime();
        Buffer.buffer_init(A, B, A_pad_rows,B_pad_cols);
        // 计算矩阵乘法
        mpi_matmul_main_process(&Buffer, C, &COMBINED_MATRIX);

        double end_time = MPI_Wtime();
        //  打印矩阵A B C
        std::cout << "Matrix A:\n";
        A.print_matrix();
        std::cout << "\nMatrix B:\n";
        B.print_matrix();
        std::cout << "\nMatrix C:\n";
        C.print_matrix();
        // 打印耗时
        std::cout << "Time taken for matrix calculation: " << end_time - start_time << " seconds\n";
    } else {
        // 从进程运算
        mpi_matmul_worker_process(&Buffer, &COMBINED_MATRIX);
    }
    MPI_Type_free(&COMBINED_MATRIX);
    MPI_Finalize();
    return 0;
}