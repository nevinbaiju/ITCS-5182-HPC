#include <immintrin.h>

void perform_addition(__m256 a, __m256 b, long int n, float result[], float temp_result[]);
void perform_addition_int(int a[], int b[], int result[], long int n);