#include "compute_kernel.h"

void perform_addition(float a[], float b[], long int n) {
    #pragma omp parallel for //num_threads(4) schedule(static, 1)
    for (int i = 0; i < n; i += 8) {
        _mm256_add_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i]));
    }
}