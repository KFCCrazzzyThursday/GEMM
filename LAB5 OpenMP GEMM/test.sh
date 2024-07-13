g++ -g -fopenmp -Wall -o OMP_Matmul main.cpp Matrix.cpp

check_output() {
    python_output=$(python output.py) 
    echo "Output of the Python script: $python_output" 
    if [[ $python_output == *"False"* ]]; then
        return 1 
    fi
    return 0 
}

for i in {1..25}; do
    echo "Iteration $i:"
    ./OMP_Matmul 16 default 0 30 30 30 > output.py
    if ! check_output; then
        echo "Detected False in the output, stopping the script."
        exit 1
    fi
done
for i in {1..25}; do
    echo "Iteration $i:"
    ./OMP_Matmul 16 static 4 30 30 30 > output.py 
    if ! check_output; then
        echo "Detected False in the output, stopping the script."
        exit 1
    fi
done
for i in {1..25}; do
    echo "Iteration $i:"
    ./OMP_Matmul 16 dynamic 4 30 30 30 > output.py
    if ! check_output; then
        echo "Detected False in the output, stopping the script."
        exit 1
    fi
done
for i in {1..25}; do
    echo "Iteration $i:"
    ./OMP_Matmul 16 guided 4 30 30 30 > output.py
    if ! check_output; then
        echo "Detected False in the output, stopping the script."
        exit 1
    fi
done
echo "Completed all iterations without detecting False."
