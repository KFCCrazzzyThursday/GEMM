#!/bin/bash

g++ -g -fopenmp -Wall -o OMP_Matmul main.cpp Matrix.cpp

# MATRIX m n k
declare -a inputs=("128 128 128" "256 256 256" "512 512 512" "1024 1024 1024" "2048 2048 2048")
# NUM THREADS
declare -a procs=(1 2 4 8 16)
# SCHEDULE
declare -a schedules=("default" "static" "dynamic" "guided")
# CHUNK SIZE
declare -a chunks=(1 4 16 64 128)
# declare -a chunks=(1)
for proc in "${procs[@]}"
do
    echo "NUM_THREAD: $proc"
    for input in "${inputs[@]}"
    do
        IFS=' ' read -r -a sizes <<< "$input"
        m=${sizes[0]}
        n=${sizes[1]}
        k=${sizes[2]}
        echo "    m, n, k: $m $n $k"
        
        for sched in "${schedules[@]}"
        do
            if [[ "$sched" == "default" ]]; then
                echo "        SCHEDULE_TYPE: $sched"
                total_time=0

                for i in {1..5}
                do
                    output=$(./OMP_Matmul $proc $sched 0 $m $n $k)
                    time_taken=$(echo "$output" | grep -o -E 'Time taken for matrix calculation: [0-9.]+ seconds' | grep -o -E '[0-9.]+')
                    total_time=$(echo "$total_time + $time_taken" | bc)
                done

                average_time=$(echo "scale=6; $total_time / 5" | bc)
                echo "        Avg time taken: $average_time seconds"
            else
                for chunk in "${chunks[@]}"
                do
                    echo "        SCHEDULE_TYPE: $sched, CHUNK_SIZE: $chunk"
                    total_time=0

                    for i in {1..5}
                    do
                        output=$(./OMP_Matmul $proc $sched $chunk $m $n $k)
                        time_taken=$(echo "$output" | grep -o -E 'Time taken for matrix calculation: [0-9.]+ seconds' | grep -o -E '[0-9.]+')
                        total_time=$(echo "$total_time + $time_taken" | bc)
                    done

                    average_time=$(echo "scale=6; $total_time / 5" | bc)
                    echo "        Avg time taken: $average_time seconds"
                done
            fi
        done
    done
done
