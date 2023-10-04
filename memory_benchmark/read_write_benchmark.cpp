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
        int tid = omp_get_thread_num();
        
        int i, allocate_status;
        int *array1, *array2;
        allocate_status = posix_memalign((void **)&array1, 32*8*4, chunk_size * 32);
        allocate_status = posix_memalign((void **)&array2, 32*8*4, chunk_size * 32);
        for(i=0; i<chunk_size; i++){
            array1[i] = 1;
        }
        
        __m256i reg_1;
        __m256i reg_2;
        __m256i reg_3;
        __m256i reg_4;
        // #pragma omp critical
        // {
        //     std::cout << "Num threads: " << num_threads << std::endl;
        //     std::cout << "TID: " << tid << " chunk size:" << chunk_size << std::endl;
        // }
        auto start = std::chrono::high_resolution_clock::now();
        for(int iter=0; iter<nbiter; iter++){ 
            for (i=0; i<chunk_size; i+=32){
                reg_1 = _mm256_stream_load_si256((__m256i *)&array1[i]);
                reg_2 = _mm256_stream_load_si256((__m256i *)&array1[i+8]);
                reg_3 = _mm256_stream_load_si256((__m256i *)&array1[i+16]);
                reg_4 = _mm256_stream_load_si256((__m256i *)&array1[i+24]);
                _mm256_store_si256((__m256i *)&array2[i], reg_1);
                _mm256_store_si256((__m256i *)&array2[i+8], reg_2);
                _mm256_store_si256((__m256i *)&array2[i+16], reg_3);
                _mm256_store_si256((__m256i *)&array2[i+24], reg_4);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        #pragma omp critical
        {
            std::chrono::duration<double> elapsed_seconds = end - start;
            seconds += elapsed_seconds.count();
            val += array2[0];
        }
        free(array1);
        free(array2);
    }
    
    std::cout << "val: " << val << std::endl;
    double read_bandwidth = (nbiter*chunk_size*num_threads*sizeof(int)) / (seconds * 1024 * 1024 * 1024); 

    std::cout << "Write Bandwidth: " << read_bandwidth << " GB/s\n";
    std::cerr << read_bandwidth << std::endl;

    return 0;
}
