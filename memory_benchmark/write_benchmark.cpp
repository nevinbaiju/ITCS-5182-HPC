#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <immintrin.h>  // AVX intrinsics

int main(int argc, char *argv[]) {
    
    if (argc < 2){
        std::cout << "Usage: ./write_benchmark <memory size (kb)>\n";
        exit(0);
    }
    int arr_size = int((std::stoi(argv[1])*1024)/4);
    std::cout << "Array size: " << (std::stoi(argv[1])*1024)/4 << std::endl;

    int *array1, *array2, *array3, *array4;
    int allocate_status;
    allocate_status = posix_memalign((void **)&array1, 32, 8 * 32);
    allocate_status = posix_memalign((void **)&array2, 32, 8 * 32);
    allocate_status = posix_memalign((void **)&array3, 32, 8 * 32);
    allocate_status = posix_memalign((void **)&array4, 32, 8 * 32);
    __m256i zero_vals = _mm256_setzero_si256();

    ////////////////////// Timing block /////////////////////////////////////////////////////
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < arr_size; i += 32) {
        _mm256_stream_si256((__m256i *)&array1[0], zero_vals);
        _mm256_stream_si256((__m256i *)&array2[0], zero_vals);
        _mm256_stream_si256((__m256i *)&array3[0], zero_vals);
        _mm256_stream_si256((__m256i *)&array4[0], zero_vals);
    }
    // for (int i = 0; i < arr_size; i ++) {
    //     array[i] = 0;
    // }
    auto end = std::chrono::high_resolution_clock::now();
    /////////////////////////////////////////////////////////////////////////////////////////

    std::chrono::duration<double> elapsed_seconds = end - start;
    double seconds = elapsed_seconds.count();   
    
    std::cout << "val1: " << array1[0] << std::endl;
    std::cout << "val2: " << array2[0] << std::endl;
    std::cout << "val3: " << array3[0] << std::endl;
    std::cout << "val4: " << array4[0] << std::endl;
    double read_bandwidth = (arr_size * sizeof(int)) / (seconds * 1024 * 1024 * 1024); // MB/s

    std::cout << "Write Bandwidth: " << read_bandwidth << " GB/s\n";
    free(array1);
    free(array2);
    free(array3);
    free(array4);

    return 0;
}
