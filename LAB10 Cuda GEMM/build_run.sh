nvcc -O3 -arch=sm_89 -maxrregcount=128 -D BLOCK_SIZE=8 -I./include test.cu utils/* kernels/* -o main -lcublas
./main 1024 1024 1024