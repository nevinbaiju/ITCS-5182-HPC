#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <immintrin.h>  // AVX intrinsics

int main(int argc, char *argv[]) {
    
    if (argc < 2){
        std::cout << "Usage: ./read_write_benchmark <memory size (kb)>\n";
        exit(0);
    }
    int arr_size = int((std::stoi(argv[1])*1024)/4);
    std::cout << "Array size: " << (std::stoi(argv[1])*1024)/4 << std::endl;

    int *arr1, *arr2;
    int allocate_status1 = posix_memalign((void **)&arr1, 32, arr_size * 32);
    int allocate_status2 = posix_memalign((void **)&arr2, 32, arr_size * 32);
    
    for (int i=0; i < arr_size; i++){
        arr1[i] = 1;
    }

    __m256i copy_data;
    ////////////////////// Timing block /////////////////////////////////////////////////////
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < arr_size; i += 8/**4*/) {
        copy_data = _mm256_stream_load_si256((__m256i *)&arr1[i]);
        _mm256_storeu_si256((__m256i *)&arr2[i], copy_data);
        // _mm256_storeu_si256((__m256i *)&array[i+8], zero_vals);
        // _mm256_storeu_si256((__m256i *)&array[i+16], zero_vals);
        // _mm256_storeu_si256((__m256i *)&array[i+24], zero_vals);
    }
    // for (int i = 0; i < arr_size; i ++) {
    //     array[i] = 0;
    // }
    auto end = std::chrono::high_resolution_clock::now();
    /////////////////////////////////////////////////////////////////////////////////////////

    std::chrono::duration<double> elapsed_seconds = end - start;
    double seconds = elapsed_seconds.count();   
    
    std::cout << "val: " << arr2[0] << std::endl;
    double read_bandwidth = (arr_size * sizeof(int)) / (seconds * 1024 * 1024 * 1024); // MB/s

    std::cout << "Read-Write Bandwidth: " << read_bandwidth << " GB/s\n";
    std::cerr << read_bandwidth << std::endl;
    
    free(arr1);
    free(arr2);

    return 0;
}
