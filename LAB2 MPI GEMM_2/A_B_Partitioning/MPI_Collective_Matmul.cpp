#include "MPI_Collective_Matmul.hpp"

// 自定义MPI派生类型
void build_mpi_mat_combined_type(const BUFFER *const buffer, MPI_Datatype *COMBINED_MATRIX)
{
    int size_A = buffer->Get_A_size_per_process();
    int size_B = buffer->Get_B_size_per_process();
    int array_of_blocklengths[2] = {size_A, size_B};
    MPI_Aint array_of_displacements[2] = {0, static_cast<MPI_Aint>(size_A * sizeof(double))};
    MPI_Datatype array_of_types[2] = {MPI_DOUBLE, MPI_DOUBLE};

    MPI_Type_create_struct(2, array_of_blocklengths, array_of_displacements, array_of_types, COMBINED_MATRIX);
    MPI_Type_commit(COMBINED_MATRIX);
}

/** MPI集合通信矩阵乘法
 * @param Buffer 缓冲区
 * @param C C_{m*k}=AB
 * @param COMBINED_MATRIX 变量类型指针
 */
void mpi_matmul_main_process(const BUFFER *const Buffer, Matrix &C, MPI_Datatype *COMBINED_MATRIX)
{
    // 计算每个进程分配的数据大小, 用于生成local_buffer
    int size_local_A = Buffer->Get_A_size_per_process();
    int size_local_B = Buffer->Get_B_size_per_process();
    // 计算每个进程算出的C的大小,用于生成local_C
    int size_local_C = Buffer->A_rows_per_process * Buffer->B_cols_per_process;
    // 计算全局合并后的comm_C的大小(在行数或列数不能整除时,可能有填充)
    int size_comm_C = Buffer->comm_size * size_local_C;
    // 计算最终结果C的大小
    int size_C = C.rows * C.cols;
    // 计算全局合并后comm_C的列数, 用于将comm_C迁移到真实的结果C中
    int comm_C_cols = Buffer->B_cols_per_process * Buffer->B_blocks;


    double *local_buffer = new double[size_local_A + size_local_B]();
    double *local_C = new double[size_local_C]();
    double *comm_C = new double[size_comm_C]();

    // 单次散射
    MPI_Scatter(Buffer->buffer, 1, *COMBINED_MATRIX, local_buffer, 1, *COMBINED_MATRIX, 0, MPI_COMM_WORLD);

    sequential_matmul(local_buffer, local_buffer + size_local_A, local_C, Buffer->A_rows_per_process, Buffer->A_cols_per_process, Buffer->B_cols_per_process);


    // 单次聚集
    MPI_Gather(local_C, size_local_C, MPI_DOUBLE, comm_C, size_local_C, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // comm_C迁移到C中
    /** C_i,j 被分到编号为 (A block_row_id, B block_col_id) 的块中,如C中第2行第2列(从0计数)的8,属于块(A1,B1)
     *  comm_C_row_id 根据 块的编号 计算 C_i,j 在 Comm_C 中的行编号, 如(A1,B1)在第 1*2+1=3行
     *  comm_C_col_id 计算 C_i,j 在 Comm_C 中的列编号,即计算 C_i,j 是其所在块的第几个元素. 如C_2,2是第(2%2*2+2%2)=0个元素
     *  因此C_2,2 = Comm_C_3,0
     */
    for (int i = 0; i < C.rows; ++i) {
        for (int j = 0; j < C.cols; ++j) {
            int block_row_id = i / Buffer->A_rows_per_process; 
            int block_col_id = j / Buffer->B_cols_per_process; 
            int comm_C_row_id = block_row_id * Buffer->B_blocks + block_col_id;
            int comm_C_col_id = (i % Buffer->A_rows_per_process) * Buffer->B_cols_per_process +
                                j % Buffer->B_cols_per_process;
            C.MAT[i * C.cols + j] =
                comm_C[comm_C_row_id * comm_C_cols + comm_C_col_id];
        }
    }
    delete[] local_buffer;
    delete[] local_C;
    delete[] comm_C;
    local_buffer = nullptr;
    local_C = nullptr;
    comm_C = nullptr;
}

/** 每个进程中计算 LOCAL_A_{localrows * n} * B_{n * k} = LOCAL_C_{localrows * k}
 */
void mpi_matmul_worker_process(const BUFFER *const Buffer, MPI_Datatype *COMBINED_MATRIX)
{
    int size_A = Buffer->Get_A_size_per_process();
    int size_B = Buffer->Get_B_size_per_process();
    int size_C = Buffer->A_rows_per_process * Buffer->B_cols_per_process;
    double *local_buffer = new double[size_A + size_B]();
    double *local_C = new double[size_C]();
    MPI_Scatter(nullptr, 0, MPI_DATATYPE_NULL, local_buffer, 1, *COMBINED_MATRIX, 0, MPI_COMM_WORLD);
    /*std::cout << "l A" << std::endl;
    print_mat(local_buffer, Buffer->A_rows_per_process, Buffer->A_cols_per_process);
    std::cout << "l B" << std::endl;
    print_mat(local_buffer + size_A, Buffer->A_cols_per_process, Buffer->B_cols_per_process);*/

    sequential_matmul(local_buffer, local_buffer + size_A, local_C, Buffer->A_rows_per_process, Buffer->A_cols_per_process, Buffer->B_cols_per_process);
    /*std::cout << "l C" << std::endl;
    print_mat(local_C, Buffer->A_rows_per_process, Buffer->B_cols_per_process);*/
    MPI_Gather(local_C, size_C, MPI_DOUBLE, nullptr, 0, MPI_DATATYPE_NULL, 0, MPI_COMM_WORLD);
    delete[] local_buffer;
    delete[] local_C;
    local_buffer = nullptr;
    local_C = nullptr;
}
