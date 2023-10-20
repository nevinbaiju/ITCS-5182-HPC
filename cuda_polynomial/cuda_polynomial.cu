#include <stdio.h>

// CUDA kernel to add 1 to every element of the array
__global__ void addOneToFloatArray(float* array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        array[idx] = idx;
    }
}

int main() {
    int n = 10;  // Size of the float array
    float* h_array = new float[n];  // Host (CPU) array

    // Initialize the host array
    for (int i = 0; i < n; i++) {
        h_array[i] = 0;
    }

    for (int i = 0; i < n; i++) {
        printf("Initialized h_array[%d] = %.2f\n", i, h_array[i]);
    }

    float* d_array;  // Device (GPU) array

    // Allocate memory on the GPU for the array
    cudaMalloc((void**)&d_array, n * sizeof(float));
    // Initialize the GPU array with zeros
    cudaMemset(d_array, 0, n * sizeof(float));

    // Copy data from the host to the device
    cudaMemcpy(d_array, h_array, n * sizeof(float), cudaMemcpyHostToDevice);

    // Define the grid and block dimensions
    int threadsPerBlock = 8;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel to add 1 to each element
    addOneToFloatArray<<<blocksPerGrid, threadsPerBlock>>>(d_array, n);

    // Copy the result back from the device to the host
    cudaMemcpy(h_array, d_array, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_array);
    delete[] h_array;

    // Print the updated array
    for (int i = 0; i < n; i++) {
        printf("Updated h_array[%d] = %.2f\n", i, h_array[i]);
    }

    return 0;
}