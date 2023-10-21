#include <stdio.h>
#include<iostream>
#include <cmath> 
#include <cuda_runtime.h>

#define checkCudaErrors(cudaCall)                                                    \
{                                                                                  \
    cudaError_t error = cudaCall;                                                  \
    if (error != cudaSuccess)                                                      \
    {                                                                              \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s\n",                \
                __FILE__, __LINE__, cudaGetErrorString(error));                     \
        exit(1);                                                                   \
    }                                                                              \
}

void validate_res(float *res, int n, int degree){
    for(int i=0; i<n; i++){
        if(std::abs(res[i]-(degree+1)) > 1){
            std::cerr << "Error: The calculation is wrong!\n";
            break;
        }
    }
}

void compute_poly(float* array, int n, float* poly, int degree, float* result){
    for(int i=0; i<n; i++){
        float x=array[i];
        float x_tothepowerof=1;
        float out=0;
        for(int j=0; j<degree; j++){
            out += poly[j]*x_tothepowerof;
            x_tothepowerof *=x;
        }
        result[i] = out;
    }
}

__global__ void compute_poly_gpu(float *array, int n, float *poly, int degree) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        float out = 0;
        float x = array[idx];
        float x_tothepowerof=1;
        for(int i=0; i<degree; i++){
            out += x_tothepowerof*poly[i];
            x_tothepowerof *= x;
        }
        array[idx] = out;
    }
}


int main() {
    int n = 10000;
    float* h_array = new float[n];
    float* h_res1 = new float[n];
    float* h_res2 = new float[n];
    
    int degree = 100;
    float* h_poly = new float[degree];
    
    for (int i = 0; i < n; i++) {
        h_array[i] = 1;
    }
    for (int i = 0; i < degree; i++) {
        h_poly[i] = 1;
    }

    float *d_array, *d_poly;
    cudaMalloc(&d_array, n*sizeof(float));
    cudaMalloc(&d_poly, degree*sizeof(float));
    cudaMemcpy(d_array, h_array, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_poly, h_poly, degree * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    compute_poly_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_array, n, d_poly, degree);
    cudaMemcpy(h_res1, d_array, n * sizeof(float), cudaMemcpyDeviceToHost);

    compute_poly(h_array, n, h_poly, degree, h_res2);
    
    
    validate_res(h_res1, n, degree);
    validate_res(h_res2, n, degree);
    
    cudaFree(d_array);
    cudaFree(d_poly);

    delete[] h_array;
    delete[] h_poly;
    delete[] h_res1;
    delete[] h_res2;
    
    return 0;
}