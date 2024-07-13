#ifndef MPI_P2P_MATMUL_HPP
#define MPI_P2P_MATMUL_HPP

#include "Matrix.hpp"
#include <mpi.h>

void mpi_matmul_main_process(const Matrix &A, const Matrix &B, Matrix &C, double &elapsed, const int comm_size);
void mpi_matmul_worker_process(const int m, const int n, const int k, const int my_rank, const int comm_size);

#endif