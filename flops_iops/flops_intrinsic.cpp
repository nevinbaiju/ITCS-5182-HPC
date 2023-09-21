#include <iostream>
#include <vector>
#include <chrono>
#include "compute_kernel.h"
#include <omp.h>

int main() {

    const long int numOperations = 1000000;
    alignas(32) float a[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    alignas(32) float b[8] = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    alignas(32) float result[8];
    alignas(32) float temp_result[8];

    __m256 _mm_a = _mm256_load_ps(&a[0]);
    __m256 _mm_b = _mm256_load_ps(&b[0]);
    __m256 _mm_result;

    int num_threads;
    #pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }
    std::cout << "Total Threads: " << num_threads << "\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    perform_addition(_mm_a, _mm_b, numOperations, result, temp_result);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    double seconds = elapsed_seconds.count();   

    // _mm256_storeu_ps(&result[0], _mm_result);
    // for(int i=0; i<8; i++){
    //     std::cout << result[i] << std::endl;
    // }

    double flops = (numOperations*8*num_threads*2/ 1e9) / seconds;
    std::cout << "Number of operations: " << numOperations*8*num_threads << std::endl;
    std::cout << "FLOPS: " << flops << " GFLOPS" << std::endl;
    std::cerr << flops << std::endl;
    
    return 0;
}
