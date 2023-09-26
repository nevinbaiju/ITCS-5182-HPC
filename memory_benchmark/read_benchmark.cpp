#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>  // AVX intrinsics

#define ARRAY_SIZE 10000000  // Adjust the array size as needed

int main() {
    int *array = (int *)malloc(ARRAY_SIZE * sizeof(int));
    if (array == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    // Initialize the array with some data
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = i;
    }

    // Measure read bandwidth using _mm256_stream_load_si256
    clock_t start_time = clock();
    unsigned int sum = 0;

    __m256i _m_sum = _mm256_setzero_si256();
    for (int i = 0; i < ARRAY_SIZE; i += 8) {
        // std::cout << i << " " << array[i] << std::endl;
        __m256i data = _mm256_loadu_si256((__m256i *)&array[i]);
        _m_sum = _mm256_add_epi32(_m_sum, data);
        int sum_array[8];
        _mm256_storeu_si256((__m256i *)sum_array, _m_sum);

        // Accumulate the sum
        for (int j = 0; j < 8; j++) {
            sum += sum_array[j];
        }
    }

    clock_t end_time = clock();

    // Calculate time taken and read bandwidth
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    double read_bandwidth = (ARRAY_SIZE * sizeof(int)) / (time_taken * 1024 * 1024); // MB/s

    printf("Read Bandwidth: %.2f MB/s\n", read_bandwidth);

    // Free allocated memory
    free(array);

    return 0;
}
