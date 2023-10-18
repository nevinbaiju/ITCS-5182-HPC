CXX := g++
CXXFLAGS_COMMON := -O3 -fopenmp
CXXFLAGS_AVX := -march=native -mavx

TARGETS := performance_modeling_avx performance_modeling_basic_parallel

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
	rm -r results
	$(CXX) $(CXXFLAGS_COMMON) $(CXXFLAGS_AVX) -D BASIC_PARALLEL=1 performance_modeling.cpp -o performance_modeling_basic_parallel
	./bench.sh performance_modeling_basic_parallel
	$(CXX) $(CXXFLAGS_COMMON) $(CXXFLAGS_AVX) -D AVX=1 performance_modeling.cpp -o performance_modeling_avx
	./bench.sh performance_modeling_avx

bench_centaurus: performance_modeling
	$(CXX) $(CXXFLAGS_COMMON) $(CXXFLAGS_AVX) -D BASIC_PARALLEL=1 performance_modeling.cpp -o performance_modeling_basic_parallel
	sbatch --partition=Centaurus --chdir=`pwd` --time=02:00:00 --ntasks=1 --cpus-per-task=16 --mem=64G --job-name=mem_1_bp bench.sh performance_modeling_basic_parallel bp
	sbatch --partition=Centaurus --chdir=`pwd` --time=04:00:00 --ntasks=1 --cpus-per-task=16 --mem=120G --job-name=mem_2_bp bench_last_res.sh performance_modeling_basic_parallel bp

	$(CXX) $(CXXFLAGS_COMMON) $(CXXFLAGS_AVX) -D AVX=1 performance_modeling.cpp -o performance_modeling_avx
	
	sbatch --partition=Centaurus --chdir=`pwd` --time=02:00:00 --ntasks=1 --cpus-per-task=16 --mem=64G --job-name=mem_1_avx bench.sh performance_modeling_avx avx
	sbatch --partition=Centaurus --chdir=`pwd` --time=04:00:00 --ntasks=1 --cpus-per-task=16 --mem=120G --job-name=mem_2_avx bench_last_res.sh performance_modeling_avx avx

clean:
	rm -f $(TARGETS)
clean_results:
	rm -rf results
	rm -rf plots