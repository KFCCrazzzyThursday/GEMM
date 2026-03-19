nvcc -O3 -arch=sm_89 -maxrregcount=128 -D BLOCK_SIZE=8 -I./include test.cu utils/* kernels/* -o test_main -lcublas
./test_main
