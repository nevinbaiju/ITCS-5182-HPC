#include <iostream>
#include <chrono>
#include <omp.h>

int main() {
    const int numOperations = 100000000;
    float result = 0.0; 

    float a = 1.0;
    float b = 2.0;
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < numOperations*8; i++) {
        a + b;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double elapsedTimeSeconds = duration.count() / 1000.0;

    double flops = (numOperations*8 / 1e9) / elapsedTimeSeconds;

    std::cout << "Number of operations: " << numOperations << std::endl;
    std::cout << "Elapsed time (seconds): " << elapsedTimeSeconds << std::endl;
    std::cout << "FLOPS: " << flops << " GFLOPS" << std::endl;

    return 0;
}