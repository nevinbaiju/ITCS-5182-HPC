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
    int nbiter = 10;

    ////////////////////// Timing block /////////////////////////////////////////////////////
    auto start = std::chrono::high_resolution_clock::now();
    __m256i _m_sum = _mm256_setzero_si256();
    for(int i=0; i<nbiter; i++)
    {
        #pragma omp parallel
        {
            // __m256i _m_sum_thread = _mm256_setzero_si256();
            // __m256i data;
            int local_sum=0;
            for (int i = 0; i < arr_size; i ++) {
                local_sum += array[i];
                // data = _mm256_stream_load_si256((__m256i *)&array[i]);
                // _m_sum_thread = _mm256_add_epi32(_m_sum_thread, data);
            }
            #pragma omp critical
            {
                // _m_sum = _mm256_add_epi32(_m_sum_thread, _m_sum);
                sum += local_sum;
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    /////////////////////////////////////////////////////////////////////////////////////////
    // _mm256_storeu_si256((__m256i *)sum_array, _m_sum);
    // for (int j = 0; j < 8; j++) {
    //         sum += sum_array[j];
    // }
    std::chrono::duration<double> elapsed_seconds = end - start;
    double seconds = elapsed_seconds.count();   
    
    std::cout << "Sum: " << sum << std::endl;
    double read_bandwidth = (nbiter*arr_size* num_threads * sizeof(int)) / (seconds * 1024 * 1024 * 1024); 

    std::cout << "Read Bandwidth: " << read_bandwidth << " GB/s\n";
    std::cerr << read_bandwidth << std::endl;

    return 0;
}
