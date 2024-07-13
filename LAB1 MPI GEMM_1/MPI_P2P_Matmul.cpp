#include "MPI_P2P_Matmul.hpp"

/** MPI点对点矩阵乘法
 * @param A A_{m*n}
 * @param B B_{n*k}
 * @param C C_{m*k}=AB
 * @param elapsed 运算时间(s)
 * @param comm_size 进程总数
 */
void mpi_matmul_main_process(const Matrix &A, const Matrix &B, Matrix &C, double &elapsed, const int comm_size)
{
    double start_time = MPI_Wtime();
    if (comm_size == 1) {
        sequential_matmul(A, B, C);
        double end_time = MPI_Wtime();
        elapsed = end_time - start_time;
        return;
    }
    // 开始计时
    int m = A.rows;
    int n = A.cols;
    int k = B.cols;
    /** 我们将A几乎平均地分给除了主进程外的其他进程，将整个B发送给每个进程
     *  从而在除主进程外的每个进程中计算 LOCAL_A_{localrows * n} * B_{n * k} = LOCAL_C_{localrows * k}
     *  主进程负责分发和拼凑LOCAL_C，组成完整的C
     *  主进程不参与分配,因此-1
     *
     * - rows_per_process 平均每个进程的行数
     * - size_per_process 平均每个进程的大小
     * - remaining_rows 未能平均分时的剩余行数(发送给最后一个进程的行数)
     * - remaining_size 未能平均分时的剩余尺寸(发送给最后一个进程的大小)
     */
    int rows_per_process = m / (comm_size - 1);
    int size_per_process = rows_per_process * n;
    int remaining_rows = m % (comm_size - 1) + rows_per_process;
    int remaining_size = remaining_rows * n;

    /** 如果该进程是主进程:
     *  那么给除了最后一个进程外的其他进程发送size_per_process的矩阵A(TAG=1)和完整矩阵B(TAG=2)
     *  给最后一个进程发送remaining_size的矩阵A和完整矩阵B
     *  然后等待接收LOCAL_C_{localrows * k}即可
     */

    // 发送阶段
    for (int process_id = 1; process_id < comm_size - 1; process_id++) {
        MPI_Send(&A.MAT[size_per_process * (process_id - 1)], size_per_process, MPI_DOUBLE, process_id, 1, MPI_COMM_WORLD);
        MPI_Send(B.MAT, n * k, MPI_DOUBLE, process_id, 2, MPI_COMM_WORLD);
    }
    MPI_Send(&A.MAT[size_per_process * (comm_size - 2)], remaining_size, MPI_DOUBLE, comm_size - 1, 1, MPI_COMM_WORLD);
    MPI_Send(B.MAT, n * k, MPI_DOUBLE, comm_size - 1, 2, MPI_COMM_WORLD);

    // 接收阶段：
    for (int process_id = 1; process_id < comm_size - 1; process_id++) {
        MPI_Recv(&C.MAT[rows_per_process * k * (process_id - 1)], rows_per_process * k, MPI_DOUBLE, process_id, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Recv(&C.MAT[rows_per_process * k * (comm_size - 2)], remaining_rows * k, MPI_DOUBLE, comm_size - 1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    double end_time = MPI_Wtime();
    elapsed = end_time - start_time;
    return;
}

/** 每个进程中计算 LOCAL_A_{localrows * n} * B_{n * k} = LOCAL_C_{localrows * k}
 *  需注意最后一个进程计算 LOCAL_A_{remainingrows * n} * B_{n * k} = LOCAL_C_{remainingrows * k}
 */
void mpi_matmul_worker_process(const int m, const int n, const int k, const int my_rank, const int comm_size)
{
    int rows_per_process = m / (comm_size - 1);
    int size_per_process = rows_per_process * n;
    int remaining_rows = m % (comm_size - 1) + rows_per_process;
    int remaining_size = remaining_rows * n;

    if (my_rank != comm_size - 1) {
        Matrix LOCAL_A(rows_per_process, n, false);
        Matrix LOCAL_B(n, k, false);
        Matrix LOCAL_C(rows_per_process, k, false);
        MPI_Recv(LOCAL_A.MAT, size_per_process, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(LOCAL_B.MAT, n * k, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        sequential_matmul(LOCAL_A, LOCAL_B, LOCAL_C);
        //std::cout << "RANK  " << my_rank << "  LOCAL_C" << std::endl;
        //LOCAL_C.print_matrix();

        MPI_Send(LOCAL_C.MAT, rows_per_process * k, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
    } else {
        Matrix LOCAL_A(remaining_rows, n, false);
        Matrix LOCAL_B(n, k, false);
        Matrix LOCAL_C(remaining_rows, k, false);
        MPI_Recv(LOCAL_A.MAT, remaining_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(LOCAL_B.MAT, n * k, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        sequential_matmul(LOCAL_A, LOCAL_B, LOCAL_C);
        //std::cout << "RANK  " << my_rank << "  LOCAL_C" << std::endl;
        //LOCAL_C.print_matrix();

        MPI_Send(LOCAL_C.MAT, remaining_rows * k, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
    }
    return;
}