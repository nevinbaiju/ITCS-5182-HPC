#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <immintrin.h>  // AVX intrinsics

int main(int argc, char *argv[]) {
    
    if (argc < 2){
        std::cout << "Usage: ./read_benchmark <memory size (kb)>\n";
        exit(0);
    }
    int arr_size = int((std::stoi(argv[1])*1024)/4);
    std::cout << "Array size: " << (std::stoi(argv[1])*1024)/4 << std::endl;

    int *array;
    int allocate_status = posix_memalign((void **)&array, 32, arr_size * 32);
    __m256i zero_vals = _mm256_setzero_si256();

    ////////////////////// Timing block /////////////////////////////////////////////////////
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < arr_size; i += 8) {
        _mm256_storeu_si256((__m256i *)&array[i], zero_vals);
    }
    // for (int i = 0; i < arr_size; i ++) {
    //     array[i] = 0;
    // }
    auto end = std::chrono::high_resolution_clock::now();
    /////////////////////////////////////////////////////////////////////////////////////////
    
    std::chrono::duration<double> elapsed_seconds = end - start;
    double seconds = elapsed_seconds.count();   
    
    std::cout << "val: " << array[0] << std::endl;
    double read_bandwidth = (arr_size * sizeof(int)) / (seconds * 1024 * 1024 * 1024); // MB/s

    std::cout << "Write Bandwidth: " << read_bandwidth << " GB/s\n";
    free(array);

    return 0;
}
