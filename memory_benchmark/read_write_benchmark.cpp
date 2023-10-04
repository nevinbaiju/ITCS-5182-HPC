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
        std::cout << "Usage: ./read_write_benchmark <memory size (kb)>\n";
        exit(0);
    }
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    int arr_size = (std::stoi(argv[1])*1024)/(sizeof(int));

    int *array1, *array2;
    int allocate_status;
    allocate_status = posix_memalign((void **)&array1, 32*8*4, arr_size * 32);
    allocate_status = posix_memalign((void **)&array2, 32*8*4, arr_size * 32);
    int nbiter = 50;
    double seconds = 0;

    for(int i=0; i<arr_size; i++){
        array1[i] = 1;
    }

    ////////////////////// Timing block /////////////////////////////////////////////////////
    
    #pragma omp parallel
    {
        int i;
        int tid = omp_get_thread_num();
        int chuck_start = tid*(arr_size/num_threads);
        int chuck_end = (tid+1)*(arr_size/num_threads);
        __m256i reg1;
        __m256i reg2;
        __m256i reg3;
        __m256i reg4;
        auto start = std::chrono::high_resolution_clock::now();
        // #pragma omp critical
        // {
        //     std::cout << "Num threads: " << num_threads << std::endl;
        //     std::cout << "TID: " << tid << ", "<< chuck_start << ":" << chuck_end << std::endl;
        // }
        for(int iter=0; iter<nbiter; iter++){ 
            for (i=chuck_start; i<chuck_end; i+=32){
                reg1 = _mm256_stream_load_si256((__m256i *)&array1[i]);
                reg2 = _mm256_stream_load_si256((__m256i *)&array1[i+8]);
                reg3 = _mm256_stream_load_si256((__m256i *)&array1[i+16]);
                reg4 = _mm256_stream_load_si256((__m256i *)&array1[i+24]);

                _mm256_storeu_si256((__m256i *)&array2[i], reg1);
                _mm256_storeu_si256((__m256i *)&array2[i+8], reg2);
                _mm256_storeu_si256((__m256i *)&array2[i+16], reg3);
                _mm256_storeu_si256((__m256i *)&array2[i+24], reg4);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        #pragma omp critical
        {
            std::chrono::duration<double> elapsed_seconds = end - start;
            seconds += elapsed_seconds.count();
        }
    }
    
    std::cout << "val: " << array2[0] << std::endl;
    double read_bandwidth = (nbiter*arr_size* num_threads * sizeof(int)) / (seconds * 1024 * 1024 * 1024); 

    std::cout << "Write Bandwidth: " << read_bandwidth << " GB/s\n";
    std::cerr << read_bandwidth << std::endl;

    free(array1);
    free(array2);

    return 0;
}
