#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using Matrix = std::vector<std::vector<double>>;

Matrix initialize_matrix(int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    Matrix matrix(rows, std::vector<double>(cols));
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
    return matrix;
}

void print_matrix(const Matrix& matrix) {
    for (const auto& row : matrix) {
        for (const auto& elem : row) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
}

Matrix matrix_multiplication(const Matrix &A, const Matrix &B, std::chrono::duration<double, std::milli>& elapsed) {
    int m = A.size(), n = A[0].size(), p = B[0].size();
    Matrix C(m, std::vector<double>(p, 0));

    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < p; ++j) {
            for(int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    return C;
}

int main(int argc, char *argv[]) {
    if(argc != 4) {
        std::cerr << "Usage: " << argv[0] << " m n k" << std::endl;
        return 1;
    }

    int m = std::stoi(argv[1]);
    int n = std::stoi(argv[2]);
    int k = std::stoi(argv[3]);

    auto A = initialize_matrix(m, n);
    auto B = initialize_matrix(n, k);
    
    std::cout << "Matrix A:\n";
    print_matrix(A);
    std::cout << "\nMatrix B:\n";
    print_matrix(B);

    std::chrono::duration<double, std::milli> elapsed;
    auto C = matrix_multiplication(A, B, elapsed);
    std::cout << "\nMatrix C:\n";
    print_matrix(C);
    std::cout << "Time taken for matrix calculation: " << elapsed.count() / 1000 << " seconds\n";
}
