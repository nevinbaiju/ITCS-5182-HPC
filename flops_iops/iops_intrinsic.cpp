#include <iostream>
#include <vector>
#include <chrono>
#include "compute_kernel.h"
#include <omp.h>

int main() {

    const long int numOperations = 1000000;
    alignas(32) int result[8];

    int num_threads;
    #pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }
    std::cout << "Total Threads: " << num_threads << "\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    perform_addition_int(numOperations, result);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    double seconds = elapsed_seconds.count();   

    double flops = (numOperations*8*num_threads/ 1e9) / seconds;
    std::cout << "Number of operations: " << numOperations*8*num_threads << std::endl;
    std::cout << "FLOPS: " << flops << " GFLOPS" << std::endl;
    std::cerr << flops << std::endl;
    
    return 0;
}
