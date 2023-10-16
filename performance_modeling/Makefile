CXX := g++
CXXFLAGS_COMMON := -O3 -fopenmp
CXXFLAGS_AVX := -march=native -mavx

TARGETS := performance_modeling

.PHONY: all clean

all: $(TARGETS) performance_modeling.cpp

performance_modeling: performance_modeling.cpp
	$(CXX) $(CXXFLAGS_COMMON) $(CXXFLAGS_AVX) -D AVX=1 performance_modeling.cpp -o performance_modeling

run: 
	./performance_modeling 10 5 3


debug:
	$(CXX) $(CXXFLAGS_COMMON) $(CXXFLAGS_AVX) -D AVX=1 -D PRINT_IMAGE=1 performance_modeling.cpp -o performance_modeling
	./performance_modeling 8 5 3

valgrind_debug:
	$(CXX) $(CXXFLAGS_COMMON) $(CXXFLAGS_AVX) -D AVX=1 -D PRINT_IMAGE=1 performance_modeling.cpp -o performance_modeling
	valgrind --log-file="debug.log" --leak-check=full --show-leak-kinds=all make run

bench: performance_modeling
	./bench.sh

bench_centaurus: performance_modeling
	sbatch --partition=Centaurus --chdir=`pwd` --time=02:00:00 --ntasks=1 --cpus-per-task=16 --mem=4G --job-name=memory_bench bench.sh

clean:
	rm -f $(TARGETS)
# clean_results:
# 	rm -rf results
# 	rm -rf plots