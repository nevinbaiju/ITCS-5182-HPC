CXX := g++
CXXFLAGS_COMMON := -O3 -fopenmp
CXXFLAGS_AVX := -march=native -mavx

SRC := read_benchmark.cpp
TARGETS := read write read_write latency latency_core_structure

.PHONY: all clean

all: $(TARGETS)

read: read_benchmark.cpp
	$(CXX) $(CXXFLAGS_COMMON) $(CXXFLAGS_AVX) read_benchmark.cpp -o read
write: write_benchmark.cpp
	$(CXX) $(CXXFLAGS_COMMON) $(CXXFLAGS_AVX) write_benchmark.cpp -o write
read_write: read_write_benchmark.cpp
	$(CXX) $(CXXFLAGS_COMMON) $(CXXFLAGS_AVX) read_write_benchmark.cpp -o read_write
latency: latency_benchmark.cpp
	$(CXX) $(CXXFLAGS_COMMON) latency_benchmark.cpp -o latency
latency_core_structure: latency_benchmark_core_structure.cpp
	$(CXX) $(CXXFLAGS_COMMON) latency_benchmark_core_structure.cpp -o latency_core_structure

run_read:
	./read 204800
run_write:
	./write 204800
run_read_write:
	./read_write 204800
run_latency:
	./latency 204800
run_latency_core_structure:
	./latency_core_structure 204800
run_all: run_read run_write run_read_write run_latency run_latency_core_structure

bench:
	./bench.sh

bench_centaurus:
	sbatch --partition=Centaurus --chdir=`pwd` --time=02:00:00 --ntasks=1 --cpus-per-task=16 --mem=4G --job-name=memory_bench bench.sh

clean:
	rm -f $(TARGETS)
clean_results:
	rm -rf results
	rm -rf plots