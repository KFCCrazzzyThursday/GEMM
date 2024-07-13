#!/bin/bash

declare -a inputs=("512 512 512" "1024 1024 1024" "1536 1536 1536" "2048 2048 2048")

for input in "${inputs[@]}"
do
    echo "对于参数m,n,k: $input"
    total_time=0

    # 每组5次
    for i in {1..5}
    do
        echo "    运行 #$i"
        output=$(python Python_Matmul.py $input)
        
        time_taken=$(echo $output | grep -o -E 'Time taken for matrix calculation: [0-9.]+ seconds' | grep -o -E '[0-9.]+')
        echo "time_taken: $time_taken"
        total_time=$(echo "$total_time + $time_taken" | bc)

    done

    # 平均
    average_time=$(echo "scale=2; $total_time / 5" | bc)
    echo "    平均运行时间: $average_time 秒"
done



