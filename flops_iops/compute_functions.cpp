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
        __m256 _mm_a1 = _mm256_setzero_ps();
        __m256 _mm_b1 = _mm256_set1_ps(2.0);
        __m256 _mm_c1 = _mm256_set1_ps(1.0);

        __m256 _mm_a2 = _mm256_setzero_ps();
        __m256 _mm_b2 = _mm256_set1_ps(2.0);
        __m256 _mm_c2 = _mm256_set1_ps(1);

        __m256 _mm_a3 = _mm256_setzero_ps();
        __m256 _mm_b3 = _mm256_set1_ps(2.0);
        __m256 _mm_c3 = _mm256_set1_ps(1);

        __m256 _mm_a4 = _mm256_setzero_ps();
        __m256 _mm_b4 = _mm256_set1_ps(2.0);
        __m256 _mm_c4 = _mm256_set1_ps(1);

        __m256 _mm_a5 = _mm256_setzero_ps();
        __m256 _mm_b5 = _mm256_set1_ps(2.0);
        __m256 _mm_c5 = _mm256_set1_ps(1.0);

        __m256 _mm_a6 = _mm256_setzero_ps();
        __m256 _mm_b6 = _mm256_set1_ps(2.0);
        __m256 _mm_c6 = _mm256_set1_ps(1);

        __m256 _mm_a7 = _mm256_setzero_ps();
        __m256 _mm_b7 = _mm256_set1_ps(2.0);
        __m256 _mm_c7 = _mm256_set1_ps(1);

        __m256 _mm_a8 = _mm256_setzero_ps();
        __m256 _mm_b8 = _mm256_set1_ps(2.0);
        __m256 _mm_c8 = _mm256_set1_ps(1);

        for (int i = 0; i < n; i +=16) {
            _mm_a1 = _mm256_fmadd_ps(_mm_a1, _mm_b1, _mm_c1);
            _mm_a2 = _mm256_fmadd_ps(_mm_a2, _mm_b2, _mm_c2);
            _mm_a3 = _mm256_fmadd_ps(_mm_a3, _mm_b3, _mm_c3);
            _mm_a4 = _mm256_fmadd_ps(_mm_a4, _mm_b4, _mm_c4);
            _mm_a5 = _mm256_fmadd_ps(_mm_a4, _mm_b4, _mm_c5);
            _mm_a6 = _mm256_fmadd_ps(_mm_a4, _mm_b4, _mm_c6);
            _mm_a7 = _mm256_fmadd_ps(_mm_a4, _mm_b4, _mm_c7);
            _mm_a8 = _mm256_fmadd_ps(_mm_a4, _mm_b4, _mm_c8);
        }
        // #pragma omp critical
        {
            _mm256_store_ps(&temp_result[0], _mm_a1);
            for(int i=0; i<8; i++){
                result[i] += temp_result[i];
            }
            _mm256_store_ps(&temp_result[0], _mm_a2);
            for(int i=0; i<8; i++){
                result[i] += temp_result[i];
            }
            _mm256_store_ps(&temp_result[0], _mm_a3);
            for(int i=0; i<8; i++){
                result[i] += temp_result[i];
            }
            _mm256_store_ps(&temp_result[0], _mm_a4);
            for(int i=0; i<8; i++){
                result[i] += temp_result[i];
            }

            _mm256_store_ps(&temp_result[0], _mm_a5);
            for(int i=0; i<8; i++){
                result[i] += temp_result[i];
            }

            _mm256_store_ps(&temp_result[0], _mm_a6);
            for(int i=0; i<8; i++){
                result[i] += temp_result[i];
            }

            _mm256_store_ps(&temp_result[0], _mm_a7);
            for(int i=0; i<8; i++){
                result[i] += temp_result[i];
            }

            _mm256_store_ps(&temp_result[0], _mm_a8);
            for(int i=0; i<8; i++){
                result[i] += temp_result[i];
            }
        }
    }
    std::cout << "Done: " << result[0] << std::endl;
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