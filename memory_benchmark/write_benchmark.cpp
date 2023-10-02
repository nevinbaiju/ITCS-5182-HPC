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
    int num_threads = omp_get_num_threads();
    int arr_size = (std::stoi(argv[1])*1024)/(sizeof(int));

    int *array;
    int allocate_status = posix_memalign((void **)&array, 32*8*4, arr_size * 32);
    int nbiter = 25;
    double seconds = 0;

    ////////////////////// Timing block /////////////////////////////////////////////////////

    #pragma omp parallel
    {
        int i;
        int tid = omp_get_thread_num();
        int chuck_start = tid*(arr_size/num_threads);
        int chuck_end = (tid+1)*(arr_size/num_threads);
        __m256i data_1 = _mm256_setzero_si256();
        __m256i data_2 = _mm256_setzero_si256();
        __m256i data_3 = _mm256_setzero_si256();
        __m256i data_4 = _mm256_setzero_si256();
        auto start = std::chrono::high_resolution_clock::now();
        for(int iter=0; iter<nbiter; iter++){ 
            for (i=chuck_start; i<chuck_end; i+=32){
                _mm256_storeu_si256((__m256i *)&array[i], data_1);
                _mm256_storeu_si256((__m256i *)&array[i+8], data_2);
                _mm256_storeu_si256((__m256i *)&array[i+16], data_3);
                _mm256_storeu_si256((__m256i *)&array[i+24], data_4);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        #pragma omp critical
        {
            std::chrono::duration<double> elapsed_seconds = end - start;
            seconds += elapsed_seconds.count();
        }
    }
    
    std::cout << "val: " << array[0] << std::endl;
    double read_bandwidth = (nbiter*arr_size* num_threads * sizeof(int)) / (seconds * 1024 * 1024 * 1024); 

    std::cout << "Read Bandwidth: " << read_bandwidth << " GB/s\n";
    std::cerr << read_bandwidth << std::endl;

    return 0;
}
