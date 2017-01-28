clean:
	rm -f *.out

gauss_gpu:		gauss.cu
	nvcc -arch=sm_50 -O2 -I/usr/local/cuda/include  gauss.cu -o gauss_gpu.out

gauss:			gauss.c
	gcc gauss.c -o gauss.out

read_matrix:	read_matrix.c
	gcc read_matrix.c -o read_matrix.out
