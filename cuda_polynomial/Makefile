CUDAFLAGS=-O3 -std=c++11
CENTFLAGS=-arch=sm_37

HOSTNAME := $(shell hostname)
DESIRED_HOSTNAME := gal-i1.uncc.edu

ifeq ($(HOSTNAME),$(DESIRED_HOSTNAME))
    TARGET := cuda_polynomial_centaurus
else
    TARGET := cuda_polynomial
endif

$(TARGET): cuda_polynomial.cu
    nvcc $(CUDAFLAGS) $(CENTFLAGS) cuda_polynomial.cu -o cuda_polynomial

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
