CUDAFLAGS=-O3 -std=c++11
CENTFLAGS=-arch=sm_37
DEBUG=0

cuda_convolution: cuda_convolution.cu 
	nvcc $(CUDAFLAGS) cuda_convolution.cu -o cuda_convolution

cuda_convolution_centaurus: cuda_convolution.cu
	nvcc $(CUDAFLAGS) $(CENTFLAGS) cuda_convolution.cu -o cuda_convolution

run:
	./cuda_convolution 10 10 3

# bench: polynomial polynomial_gpu
# 	./queue_cpu.sh
# 	./queue_gpu.sh

# test: polynomial_gpu polynomial
# 	./queue_test_cpu.sh
# 	./queue_test_gpu.sh

# table:
# 	./table.sh

clean:
	-rm cuda_convolution
