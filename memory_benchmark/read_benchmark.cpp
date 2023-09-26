#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <immintrin.h>  // AVX intrinsics

#define ARRAY_SIZE (200*(1024*1024))/4  // Adjust the array size as needed

int main() {
    int *array;
    int allocate_status = posix_memalign((void **)&array, 32, 8 * 32);
    for(int i=0; i<8; i++){
        array[i] = i;
    }
    unsigned int sum = 0;
    __m256i _m_sum = _mm256_setzero_si256();
    int sum_array[8];
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ARRAY_SIZE; i += 8) {
        __m256i data = _mm256_stream_load_si256((__m256i *)&array[0]);
        _m_sum = _mm256_add_epi32(_m_sum, data);
        _mm256_storeu_si256((__m256i *)sum_array, _m_sum);
    
        for (int j = 0; j < 8; j++) {
            sum += sum_array[j];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    double seconds = elapsed_seconds.count();   
    
    std::cout << "Sum: " << sum << std::endl;
    double read_bandwidth = (ARRAY_SIZE * sizeof(int)) / (seconds * 1024 * 1024 * 1024); // MB/s

    std::cout << "Read Bandwidth: " << read_bandwidth << " GB/s\n";

    return 0;
}
