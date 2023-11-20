CUDAFLAGS=-O3 -std=c++11
CENTFLAGS=-arch=sm_37
DEBUG=0

pipelined_polynomial: pipelined_polynomial.cu 
	nvcc $(CUDAFLAGS) pipelined_polynomial.cu -o pipelined_polynomial

pipelined_polynomial_centaurus: pipelined_polynomial.cu
	nvcc $(CUDAFLAGS) $(CENTFLAGS) pipelined_polynomial.cu -o pipelined_polynomial

pipelined_polynomial_multi: pipelined_polynomial_multi_gpu.cu
	nvcc $(CUDAFLAGS) $(CENTFLAGS) pipelined_polynomial_multi_gpu.cu -o pipelined_polynomial_m

run:
	./pipelined_polynomial

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
