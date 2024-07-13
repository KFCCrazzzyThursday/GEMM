nvcc -O3 -arch=sm_89 --ptxas-options=-v -maxrregcount=128 -D BLOCK_SIZE=8 -I./include test.cu utils/* kernels/* -o test_8 -lcublas
./test_8 > test_block_8.txt
rm ./test_8

nvcc -O3 -arch=sm_89 --ptxas-options=-v -maxrregcount=128 -D BLOCK_SIZE=16 -I./include test.cu utils/* kernels/* -o test_16 -lcublas
./test_16 > test_block_16.txt
rm ./test_16