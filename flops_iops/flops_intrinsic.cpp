#include <iostream>
#include <vector>
#include <chrono>
#include "compute_kernel.h"

int main() {

    const long int numOperations = 8*1000000;
    float *a = new float[numOperations];
    float *b = new float[numOperations];
    float *result = new float[numOperations];

    for(int i=0; i<numOperations; i++){
        a[i] = 1.0;
        b[i] = 2.0;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    perform_addition(a, b, result, numOperations);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    double seconds = elapsed_seconds.count();   
    // for(int i=numOperations-20; i<numOperations; i++){
    //     std::cout << result[i] << std::endl;
    // }

    double flops = (numOperations/ 1e9) / seconds;
    std::cout << "Number of operations: " << numOperations << std::endl;
    std::cout << "FLOPS: " << flops << " GFLOPS" << std::endl;
    std::cerr << flops << std::endl;
    
    return 0;
}
