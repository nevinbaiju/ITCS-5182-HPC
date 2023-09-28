#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <immintrin.h>  // AVX intrinsics
#include <string>
#include <omp.h>

// #define ARRAY_SIZE (2000*(1024*1024))/4  // Adjust the array size as needed

int main(int argc, char *argv[]) {
    
    if (argc < 2){
        std::cout << "Usage: ./read_benchmark <memory size (kb)>\n";
        exit(0);
    }
    int arr_size = (std::stoi(argv[1])*1024)/sizeof(int);

    int *array;
    int allocate_status = posix_memalign((void **)&array, 32, arr_size * 32);
    for(int i=0; i<8; i++){
        array[i] = i;
    }
    unsigned int sum = 0;
    __m256i _m_sum = _mm256_setzero_si256();
    __m256i data;
    int sum_array[8];

    for (int i = 0; i < arr_size; i += 8) {
        data = _mm256_stream_load_si256((__m256i *)&array[0]);
        _m_sum = _mm256_add_epi32(_m_sum, data);
    }
    ////////////////////// Timing block /////////////////////////////////////////////////////
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp for
    for (int i = 0; i < arr_size; i += 8) {
        data = _mm256_stream_load_si256((__m256i *)&array[0]);
        _m_sum = _mm256_add_epi32(_m_sum, data);
    }
    auto end = std::chrono::high_resolution_clock::now();
    /////////////////////////////////////////////////////////////////////////////////////////
    _mm256_storeu_si256((__m256i *)sum_array, _m_sum);
    for (int j = 0; j < 8; j++) {
            sum += sum_array[j];
    }
    std::chrono::duration<double> elapsed_seconds = end - start;
    double seconds = elapsed_seconds.count();   
    
    std::cout << "Sum: " << sum << std::endl;
    double read_bandwidth = (arr_size * sizeof(int)) / (seconds * 1024 * 1024 * 1024); 

    std::cout << "Read Bandwidth: " << read_bandwidth << " GB/s\n";

    return 0;
}
