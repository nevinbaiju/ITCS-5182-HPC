CXX := g++
CXXFLAGS_COMMON := -O3 -fopenmp
CXXFLAGS_AVX := -march=native -mavx

TARGETS := performance_modeling

.PHONY: all clean

all: $(TARGETS) performance_modeling.cpp

performance_modeling: performance_modeling.cpp
	$(CXX) performance_modeling.cpp -o performance_modeling

run: 
	./performance_modeling 10 5

debug:
	valgrind --log-file="debug.log" --leak-check=full --show-leak-kinds=all ./performance_modeling 10 5 

# bench: read
# 	./bench.sh

# bench_centaurus: read
# 	sbatch --partition=Centaurus --chdir=`pwd` --time=02:00:00 --ntasks=1 --cpus-per-task=16 --mem=4G --job-name=memory_bench bench.sh

clean:
	rm -f $(TARGETS)
# clean_results:
# 	rm -rf results
# 	rm -rf plots