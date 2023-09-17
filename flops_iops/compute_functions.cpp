#include "compute_kernel.h"
#include <iostream>

void perform_addition(float a[], float b[], float result[], long int n) {
    #pragma omp parallel for //num_threads(4) schedule(static, 1)
    for (int i = 0; i < n; i += 8) {
        _mm256_add_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i]));
        // __m256 temp_result = _mm256_add_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i]));
        // _mm256_storeu_ps(&result[i], temp_result);
    }
}

void perform_addition_int(int a[], int b[], int result[], long int n) {
    #pragma omp parallel for //num_threads(4) schedule(static, 1)
    for (int i = 0; i < n; i += 8) {
        _mm256_add_epi32(_mm256_loadu_si256((__m256i*)&a[i]), 
                         _mm256_loadu_si256((__m256i*)&b[i]));
    }
}