import random
import time
import sys

def initialize_matrix(rows, cols):
    """随机初始化"""
    return [[random.random() for _ in range(cols)] for _ in range(rows)]

def matrix_multiplication(A, B):
    """basic triple loop"""
    m, n = len(A), len(A[0])
    p = len(B[0])
    C = [[0 for _ in range(p)] for _ in range(m)]
    
    start_time = time.time()  # tik
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    end_time = time.time()  # tok
    return C, end_time - start_time

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python Python_Matmul.py m n k")
        sys.exit(1)
    
    m, n, k = map(int, sys.argv[1:4])
    assert 512 <= m <= 2048, "m must be \in [512, 2048]"
    assert 512 <= n <= 2048, "n must be \in [512, 2048]"
    assert 512 <= k <= 2048, "k must be \in [512, 2048]"
    A = initialize_matrix(m, n)
    B = initialize_matrix(n, k)
    
    C, time_taken = matrix_multiplication(A, B)
    print(A)
    print(B)
    print(C)
    print(f"Time taken for matrix calculation: {time_taken} seconds")
    
