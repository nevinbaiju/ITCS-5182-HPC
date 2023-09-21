#include "compute_kernel.h"
#include <iostream>
#include <omp.h>

// void perform_addition(float a[], float b[], float result[], long int n) {
//     #pragma omp parallel for num_threads(4)// schedule(static, 1)
//     for (int i = 0; i < n; i += 8) {
//         _mm256_add_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i]));
//         // __m256 temp_result = _mm256_add_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i]));
//         // _mm256_storeu_ps(&result[i], temp_result);
//     }
// }

// void perform_addition(__m256 a, __m256 b, long int n, float result[], float temp_result[]) {
//     __m256 _mm_temp_result;
//     #pragma omp parallel
//     {
//         for (int i = 0; i < n; i ++) {
//             a = _mm256_add_ps(a, b);
//             // __m256 temp_result = _mm256_add_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i]));
//             // _mm256_storeu_ps(&result[i], temp_result);
//         }
//         #pragma omp critical
//         {
//             _mm256_store_ps(&temp_result[0], _mm_temp_result);
//             result[0] += temp_result[0];
//             // for(int i=0; i<8; i++){
//             //     result[i] += temp_result[i];
//             // }
//         }
//     }
//     std::cout << "Done: " << result[0] << std::endl;
//     // for(int i=0; i<8; i++){
//     //     std::cout << result[i] << ", " << std::endl;
//     // }
    
//     // return a;
// }

void perform_addition(__m256 a, __m256 b, long int n, float result[], float temp_result[]) {
    #pragma omp parallel
    {
        __m256 _mm_a = _mm256_setzero_ps();
        __m256 _mm_b = _mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);

        __m256 _mm_c = _mm256_setzero_ps();
        __m256 _mm_d = _mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        for (int i = 0; i < n; i +=2) {
            _mm_a = _mm256_add_ps(_mm_a, _mm_b);
            _mm_c = _mm256_mul_ps(_mm_c, _mm_d);
        }
        #pragma omp critical
        {
            _mm256_store_ps(&temp_result[0], _mm_a);
            // result[0] += temp_result[0];
            for(int i=0; i<8; i++){
                result[i] += temp_result[i];
            }
        }
    }
    std::cout << "Done: " << result[0] << std::endl;
    // for(int i=0; i<8; i++){
    //     std::cout << result[i] << ", " << std::endl;
    // }
    
    // return a;
}

void perform_addition_int(int a[], int b[], int result[], long int n) {
    #pragma omp parallel for //num_threads(4) schedule(static, 1)
    for (int i = 0; i < n; i += 8*8) {
        for (int j = 0; j < 8; ++j) {
            _mm256_add_epi32(_mm256_loadu_si256((__m256i*)&a[i+j*8]), 
                            _mm256_loadu_si256((__m256i*)&b[i+j*8]));
        }
    }
}