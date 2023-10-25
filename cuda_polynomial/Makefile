CUDAFLAGS=-O3 -std=c++11

cuda_polynomial: cuda_polynomial.cu
	nvcc $(CUDAFLAGS) cuda_polynomial.cu -o cuda_polynomial

run:
	./cuda_polynomial 100000 100

# bench: polynomial polynomial_gpu
# 	./queue_cpu.sh
# 	./queue_gpu.sh

# test: polynomial_gpu polynomial
# 	./queue_test_cpu.sh
# 	./queue_test_gpu.sh

# table:
# 	./table.sh

clean:
	-rm polynomial polynomial_gpu
