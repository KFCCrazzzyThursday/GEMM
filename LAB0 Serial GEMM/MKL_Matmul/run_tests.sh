#!/bin/bash

g++ -o MKL_Matmul MKL_Matmul.cpp -I"$MKLROOT/include" -L"$MKLROOT/lib/intel64" -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
declare -a inputs=("512 512 512" "1024 1024 1024" "1536 1536 1536" "2048 2048 2048")

for input in "${inputs[@]}"
do
    echo "对于参数m,n,k: $input"
    total_time=0

    # 每组运行5次
    for i in {1..5}
    do
        echo "    运行 #$i"
        # 捕获输出
        output=$(MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 ./MKL_Matmul $input)
        
        time_taken=$(echo "$output" | grep -o -E 'Time taken for matrix calculation: [0-9.]+ seconds' | grep -o -E '[0-9.]+')
        echo "time_taken: $time_taken"
        total_time=$(echo "$total_time + $time_taken" | bc)
    done

    average_time=$(echo "scale=2; $total_time / 5" | bc)
    echo "    平均运行时间: $average_time 秒"
done
