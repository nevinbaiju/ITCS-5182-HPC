#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <immintrin.h>  // AVX intrinsics

#define ARRAY_SIZE (200*(1024*1024))/4  // Adjust the array size as needed

int main() {
    int *array;
    int n=32000000;
    int allocate_status = posix_memalign((void **)&array, 32, n * 32);
    __m256i zero_vals = _mm256_setzero_si256();
    auto start = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < ARRAY_SIZE; i += 8) {
    //     _mm256_storeu_si256((__m256i *)&array[i], zero_vals);
    // }
    for (int i = 0; i < ARRAY_SIZE; i ++) {
        array[i] = 0;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    double seconds = elapsed_seconds.count();   
    
    std::cout << "val: " << array[0] << std::endl;
    double read_bandwidth = (ARRAY_SIZE * sizeof(int)) / (seconds * 1024 * 1024 * 1024); // MB/s

    std::cout << "Write Bandwidth: " << read_bandwidth << " GB/s\n";

    return 0;
}
