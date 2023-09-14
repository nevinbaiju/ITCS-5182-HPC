#include <iostream>
#include <immintrin.h>
#include <vector>
#include <chrono>

void perform_addition(float a[], float b[], long int n){
    #pragma omp parallel for //num_threads(4) schedule(static, 1)
    for (int i = 0; i < n; i++) {
        _mm256_add_ps(_mm256_load_ps(&a[0]), _mm256_load_ps(&b[0]));
    }
}

int main() {

    const long int numOperations = 1000000000;
    alignas(32) float a[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    alignas(32) float b[8] = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    
    auto start = std::chrono::high_resolution_clock::now();
    perform_addition(a, b, numOperations);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    double seconds = elapsed_seconds.count();

    double flops = (numOperations *8/ 1e9) / seconds;
    std::cout << "Number of operations: " << numOperations << std::endl;
    std::cout << "FLOPS: " << flops << " GFLOPS" << std::endl;
    
    return 0;
}
