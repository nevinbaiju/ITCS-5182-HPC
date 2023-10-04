#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <immintrin.h>
#include <string>
#include <omp.h>

int main(int argc, char *argv[]) {
    
    if (argc < 2){
        std::cout << "Usage: ./read_benchmark <memory size (kb)>\n";
        exit(0);
    }
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    int chunk_size = (std::stoi(argv[1])*1024)/(sizeof(int)*num_threads);
    int nbiter = 25;
    double seconds = 0;
    int val = 0;
    
    #pragma omp parallel
    {
        int i;
        int tid = omp_get_thread_num();
        int *array;
        int allocate_status = posix_memalign((void **)&array, 32*8*4, chunk_size * 32);
        __m256i data_1 = _mm256_setzero_si256();
        __m256i data_2 = _mm256_setzero_si256();
        __m256i data_3 = _mm256_setzero_si256();
        __m256i data_4 = _mm256_setzero_si256();
        // #pragma omp critical
        // {
        //     std::cout << "Num threads: " << num_threads << std::endl;
        //     std::cout << "TID: " << tid << " chunk size:" << chunk_size << std::endl;
        // }
        auto start = std::chrono::high_resolution_clock::now();
        for(int iter=0; iter<nbiter; iter++){ 
            for (i=0; i<chunk_size; i+=32){
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
            val += array[0];
        }
        free(array);
    }
    
    std::cout << "val: " << val << std::endl;
    double read_bandwidth = (nbiter*chunk_size*num_threads*sizeof(int)) / (seconds * 1024 * 1024 * 1024); 

    std::cout << "Write Bandwidth: " << read_bandwidth << " GB/s\n";
    std::cerr << read_bandwidth << std::endl;

    return 0;
}
