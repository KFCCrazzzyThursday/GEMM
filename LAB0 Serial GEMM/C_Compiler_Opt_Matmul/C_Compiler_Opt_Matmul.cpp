#include <iostream>
#include <chrono>
#include <cstdlib>

double** initialize_matrix(int rows, int cols) {
    double** matrix = new double*[rows];
    for(int i = 0; i < rows; ++i) {
        matrix[i] = new double[cols];
        for(int j = 0; j < cols; ++j) {
            matrix[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }
    }
    return matrix;
}

void matrix_multiplication(int m, int n, int p, double** __restrict__ A, double** __restrict__ B, double** __restrict__ C) {
    for(int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j){
                for (int k = 0; k < n; ++k) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
    }
}

int main(int argc, char *argv[]) {
    if(argc != 4) {
        std::cerr << "Usage: " << argv[0] << " m n k" << std::endl;
        return 1;
    }

    int m = std::stoi(argv[1]), n = std::stoi(argv[2]), k = std::stoi(argv[3]);

    double** A = initialize_matrix(m, n);
    double** B = initialize_matrix(n, k);
    double** C = initialize_matrix(m, k); // Initialize C with zeros

    auto start = std::chrono::high_resolution_clock::now();
    matrix_multiplication(m, n, k, A, B, C);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Time taken for matrix calculation: " << elapsed.count() / 1000 << " seconds\n";

    for(int i = 0; i < m; ++i) {
        delete[] A[i];
        delete[] C[i];
    }
    delete[] A;
    delete[] C;

    for(int i = 0; i < n; ++i) {
        delete[] B[i];
    }
    delete[] B;
}
