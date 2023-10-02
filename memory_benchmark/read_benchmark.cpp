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
    int allocate_status = posix_memalign((void **)&array, 32, arr_size * 32);
    for(int i=0; i<arr_size; i++){
        array[i] = 1;
    }
    int sum = 0;
    int sum_array[8];
    int nbiter = 25;
    double seconds = 0;

    ////////////////////// Timing block /////////////////////////////////////////////////////
    
    __m256i _m_sum = _mm256_setzero_si256();
    // #pragma omp parallel
    // {
    //     auto start = std::chrono::high_resolution_clock::now();
    //     int local_sum=0, i;
    //     for(int iter=0; iter<nbiter; iter++){ 
    //         for (i=0; i<arr_size; i++){
    //             local_sum += array[i];
    //         }
    //     }
    //     auto end = std::chrono::high_resolution_clock::now();
    //     #pragma omp critical
    //     {
    //         std::chrono::duration<double> elapsed_seconds = end - start;
    //         seconds += elapsed_seconds.count();
    //         sum += local_sum;
    //     }
    // }

    #pragma omp parallel
    {
        auto start = std::chrono::high_resolution_clock::now();
        int i;
        __m256i data1, data2, data3, data4;
        __m256i _m_sum_local_1 = _mm256_setzero_si256();
        __m256i _m_sum_local_2 = _mm256_setzero_si256();
        __m256i _m_sum_local_3 = _mm256_setzero_si256();
        __m256i _m_sum_local_4 = _mm256_setzero_si256();

        for(int iter=0; iter<nbiter; iter++){ 
            for (i=0; i<arr_size; i+=32){
                data1 = _mm256_stream_load_si256((__m256i *)&array[i]);
                data2 = _mm256_stream_load_si256((__m256i *)&array[i+8]);
                data3 = _mm256_stream_load_si256((__m256i *)&array[i+16]);
                data4 = _mm256_stream_load_si256((__m256i *)&array[i+24]);
                
                _m_sum_local_1 = _mm256_add_epi32(_m_sum_local_1, data1);
                _m_sum_local_2 = _mm256_add_epi32(_m_sum_local_2, data2);
                _m_sum_local_3 = _mm256_add_epi32(_m_sum_local_3, data3);
                _m_sum_local_4 = _mm256_add_epi32(_m_sum_local_4, data4);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        #pragma omp critical
        {
            std::chrono::duration<double> elapsed_seconds = end - start;
            seconds += elapsed_seconds.count();
            _m_sum = _mm256_add_epi32(_m_sum, _m_sum_local_1);
            _m_sum = _mm256_add_epi32(_m_sum, _m_sum_local_2);
            _m_sum = _mm256_add_epi32(_m_sum, _m_sum_local_3);
            _m_sum = _mm256_add_epi32(_m_sum, _m_sum_local_4);
        }
    }
    _mm256_storeu_si256((__m256i *)sum_array, _m_sum);
    /////////////////////////////////////////////////////////////////////////////////////////
    // _mm256_storeu_si256((__m256i *)sum_array, _m_sum);
    // for (int j = 0; j < 8; j++) {
    //         sum += sum_array[j];
    // }
    
    std::cout << "Sum: " << sum_array[0] << std::endl;
    double read_bandwidth = (nbiter*arr_size* num_threads * sizeof(int)) / (seconds * 1024 * 1024 * 1024); 

    std::cout << "Read Bandwidth: " << read_bandwidth << " GB/s\n";
    std::cerr << read_bandwidth << std::endl;

    return 0;
}
