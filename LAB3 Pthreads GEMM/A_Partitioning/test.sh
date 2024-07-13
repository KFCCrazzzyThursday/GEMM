#!/bin/bash

g++ -g -Wall -o A_Partitioning_Pthread_Matmul main.cpp Matrix.cpp Pthread_Matmul.cpp -lpthread

# 矩阵维度
declare -a inputs=("128 128 128" "256 256 256" "512 512 512" "1024 1024 1024" "2048 2048 2048")
# 进程数量
declare -a procs=(1 2 4 8 16)

for proc in "${procs[@]}"
do
    echo "worker线程数量: $((proc))"
    for input in "${inputs[@]}"
    do
        echo "    对于参数m,n,k: $input"
        total_time=0

        # 每组运行3次
        for i in {1..3}
        do
            echo "        运行 #$i"
            # 使用mpiexec运行
            output=$(./A_Partitioning_Pthread_Matmul $proc $input)
            
            time_taken=$(echo "$output" | grep -o -E 'Time taken for matrix calculation: [0-9.]+ seconds' | grep -o -E '[0-9.]+')
            echo "        time_taken: $time_taken"
            total_time=$(echo "$total_time + $time_taken" | bc)
        done

        average_time=$(echo "scale=6; $total_time / 3" | bc)
        echo "        平均运行时间: $average_time 秒"
    done
done
