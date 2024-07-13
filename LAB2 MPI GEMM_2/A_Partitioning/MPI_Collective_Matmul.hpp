#ifndef MPI_Collective_MATMUL_HPP
#define MPI_Collective_MATMUL_HPP

#include "Matrix.hpp"
#include "buffer.hpp"


// 自定义MPI派生类型
void build_mpi_mat_combined_type(const BUFFER *const buffer, MPI_Datatype *COMBINED_MATRIX);

void mpi_matmul_main_process(const BUFFER *const buffer, Matrix &C, MPI_Datatype *COMBINED_MATRIX);
void mpi_matmul_worker_process(const BUFFER *const buffer,MPI_Datatype *COMBINED_MATRIX);

#endif