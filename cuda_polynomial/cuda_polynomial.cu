#include <stdio.h>
#include<iostream>
#include <cmath> 
#include <cuda_runtime.h>
#include <chrono>

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

double get_time_elapsed(std::chrono::time_point<std::chrono::high_resolution_clock> start, std::chrono::time_point<std::chrono::high_resolution_clock> end){
    std::chrono::duration<double> elapsed_seconds = end - start;
    double seconds = elapsed_seconds.count();

    return seconds;
}

void validate_res(float *res, int n, int degree){
    for(int i=0; i<n; i++){
        if(std::abs(res[i]-(degree+1)) > 1){
            std::cerr << "Error: The calculation is wrong!\n";
            break;
        }
    }
    std::cout << "Polynomials computed succesfully!\n";
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


int main(int argc, char *argv[]) {
    int n = atoi(argv[1]);
    int degree = atoi(argv[2]);

    float* h_array = new float[n];
    float* h_res1 = new float[n];
    float* h_res2 = new float[n];
    
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

    auto start_memcpy =  std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_array, h_array, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_poly, h_poly, degree * sizeof(float), cudaMemcpyHostToDevice);
    auto end_memcpy =  std::chrono::high_resolution_clock::now();

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    auto start_compute = std::chrono::high_resolution_clock::now();
    
    compute_poly_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_array, n, d_poly, degree);
    cudaDeviceSynchronize();
    
    auto end_compute = std::chrono::high_resolution_clock::now();
    
    cudaMemcpy(h_res1, d_array, n * sizeof(float), cudaMemcpyDeviceToHost);

    compute_poly(h_array, n, h_poly, degree, h_res2);
    
    
    validate_res(h_res1, n, degree);
    // validate_res(h_res2, n, degree);

    double compute_time = get_time_elapsed(start_compute, end_compute);
    double flop = (n/1e9)*3*(degree + 1);
    double flops =  flop/(compute_time*1e3);
    double mem_bw = ((n/1e6)*4)/(compute_time*1e3);
    std::cout << "FLOPS: " << flops << " Terra FLOPS" << std::endl;
    std::cout << "GPU Memory Bandwidth: " << mem_bw << " GB/s" << std::endl;

    double pci_bandwidth_time = get_time_elapsed(start_memcpy, end_memcpy);
    double pci_bandwidth = ((n/1e6)*4)/(1e3*pci_bandwidth_time);
    std::cout << "PCI-e Latency: " << pci_bandwidth_time << " seconds" << std::endl;
    std::cout << "PCI-e Bandwidth: " << pci_bandwidth << " GB/s" << std::endl;

    std::cerr << n << "," << degree << "," << flops << "," << mem_bw << "," << pci_bandwidth_time << "," << pci_bandwidth << "\n";
    
    cudaFree(d_array);
    cudaFree(d_poly);

    delete[] h_array;
    delete[] h_poly;
    delete[] h_res1;
    delete[] h_res2;
    
    return 0;
}