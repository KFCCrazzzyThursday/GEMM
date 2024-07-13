#include <iostream>
#include "mkl.h"
#include <chrono>

double *initialize_matrix(int rows, int cols)
{
    double *matrix = (double *)mkl_malloc(rows * cols * sizeof(double), 64);
    if (matrix == NULL) {
        std::cerr << "Memory allocation failed" << std::endl;
        exit(1);
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = static_cast<double>(rand()) / RAND_MAX;
        }
    }
    return matrix;
}

void print_matrix(const char *name, double *matrix, int rows, int cols)
{
    std::cout << "Matrix " << name << ":" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " m n k" << std::endl;
        return 1;
    }

    int m = std::stoi(argv[1]), n = std::stoi(argv[2]), k = std::stoi(argv[3]);
    double alpha = 1.0, beta = 0.0;

    double *A = initialize_matrix(m, k);
    double *B = initialize_matrix(k, n);
    double *C = initialize_matrix(m, n);
    auto start = std::chrono::high_resolution_clock::now();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C, n);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    

    print_matrix("A", A, m, k);
    print_matrix("B", B, k, n);
    print_matrix("C", C, m, n);
    std::cout << "Time taken for matrix calculation: " << elapsed.count() << " seconds\n";
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    return 0;
}
