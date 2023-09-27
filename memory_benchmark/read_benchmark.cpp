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
        std::cout << "Usage: ./read_benchmark <memory size (mb)>\n";
        exit(0);
    }
    int mem_size = std::stoi(argv[1]);

    int *array;
    int allocate_status = posix_memalign((void **)&array, 32, 8 * 32);
    for(int i=0; i<8; i++){
        array[i] = i;
    }
    unsigned int sum = 0;
    __m256i _m_sum = _mm256_setzero_si256();
    __m256i data;
    int sum_array[8];

    ////////////////////// Timing block /////////////////////////////////////////////////////
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp for
    for (int i = 0; i < mem_size; i += 8) {
        data = _mm256_stream_load_si256((__m256i *)&array[0]);
        _m_sum = _mm256_add_epi32(_m_sum, data);
        _mm256_storeu_si256((__m256i *)sum_array, _m_sum);
    
        for (int j = 0; j < 8; j++) {
            sum += sum_array[j];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    /////////////////////////////////////////////////////////////////////////////////////////

    std::chrono::duration<double> elapsed_seconds = end - start;
    double seconds = elapsed_seconds.count();   
    
    std::cout << "Sum: " << sum << std::endl;
    double read_bandwidth = (mem_size * sizeof(int)) / (seconds * 1024 * 1024 * 1024); // MB/s

    std::cout << "Read Bandwidth: " << read_bandwidth << " GB/s\n";

    return 0;
}
