#include <stdio.h>
#include<iostream>
#include <cmath> 
#include <cuda_runtime.h>
#include <iomanip>
#include <chrono>
#include <cstring>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

double get_time_elapsed(std::chrono::time_point<std::chrono::high_resolution_clock> start, std::chrono::time_point<std::chrono::high_resolution_clock> end){
    std::chrono::duration<double> elapsed_seconds = end - start;
    double seconds = elapsed_seconds.count();

    return seconds;
}

void validate_res(float *res, int n, int degree){
    for(int i=0; i<n; i++){
        if(std::abs(res[i]-(degree+1)) > 1){
            std::cout << "Error: The calculation is wrong!\n";
            break;
        }
    }
    std::cout << "Polynomials computed succesfully!\n";
}

__global__ void convolve(float *image, float *res_image, float *filter, int height, int width, int filter_size, int padded_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < height*width){
        float res = 0;
        int y = int(idx/width), x = idx%width;
        for(int i=y; i<y+filter_size; i++){
            for(int j=x; j<x+filter_size; j++){
                res += image[i*padded_width + j]*filter[(i-y)*filter_size + (j-x)];
            }
        }
        res_image[idx] = res;
    }
}

void init_image(float *image, int width, int height, int padding){
    int pixel_val = 1;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            // Account for padding.
            if ((y<padding)|(x<padding)|(y>=(height-padding))|(x>=(width-padding))){
                image[y*height + x] = 0;
                continue;   
            }
            else{
                    image[y*height + x] = pixel_val;
                    pixel_val++;
            }
        }
    }
}

void generate_identity_kernel(float *&filter, int filter_size){
    for(int i=0; i<filter_size; i++){
        for(int j=0; j<filter_size; j++){
            filter[i*filter_size + j] = 0;
        }
    }
    int mid_point = int(filter_size/2);
    filter[mid_point*filter_size + mid_point] = 1;
}

void print_image(float *image, int width, int height){
    for (int y=0; y<height; y++){
        for (int x=0; x<width; x++){
            std::cout << std::setw(5) << std::setfill(' ') << image[y*height + x] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}


int main(int argc, char *argv[]) {
    int height = atoi(argv[1]);
    int width = atoi(argv[2]);
    int filter_size = atoi(argv[3]);

    int padding = 2*int(filter_size/2);

    float* h_image = new float[(height+padding)*(width+padding)];
    float* h_result = new float[height*width];
    float* h_filter = new float[filter_size*filter_size];

    init_image(h_image, width+padding, height+padding, padding/2);
    generate_identity_kernel(h_filter, filter_size);

    // print_image(h_filter, filter_size, filter_size);
    // print_image(h_image, width+padding, height+padding);

    float *d_image, *d_filter, *d_result;
    gpuErrchk(cudaMalloc(&d_image, (height+padding)*(width+padding)*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_filter, filter_size*filter_size*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_result, height*width*sizeof(float)));
    
    gpuErrchk(cudaMemcpy(d_image, h_image, (height+padding)*(width+padding)*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_filter, h_filter, filter_size*filter_size*sizeof(float), cudaMemcpyHostToDevice));

    int threadsPerBlock = 512;
    int blocksPerGrid = (height*width + threadsPerBlock - 1) / threadsPerBlock;
    
    // auto start_compute = std::chrono::high_resolution_clock::now();
    
    // for(int i=0; i<nb_iter; i++){
        convolve<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_result, d_filter, height, width, filter_size, width+padding);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError() );
    // }
    
    // auto end_compute = std::chrono::high_resolution_clock::now();
    
    gpuErrchk(cudaMemcpy(h_result, d_result, height*width*sizeof(float), cudaMemcpyDeviceToHost));
    // print_image(h_result, width, height);
    
    // validate_res(h_res1, n, degree);
    // // validate_res(h_res2, n, degree);

    // double compute_time = get_time_elapsed(start_compute, end_compute);
    // double flop = ((nb_iter*n)/1e9)*3*(degree + 1);
    // double flops =  flop/(compute_time*1e3);
    // double mem_bw = (((nb_iter*n)/1e6)*4)/(compute_time*1e3);
    // std::cout << "FLOPS: " << flops << " Terra FLOPS" << std::endl;
    // std::cout << "GPU Memory Bandwidth: " << mem_bw << " GB/s" << std::endl;

    // double pci_bandwidth_time = get_time_elapsed(start_memcpy, end_memcpy);
    // double pci_bandwidth = ((n/1e6)*4)/(1e3*pci_bandwidth_time);
    // std::cout << "PCI-e Latency: " << pci_bandwidth_time << " seconds" << std::endl;
    // std::cout << "PCI-e Bandwidth: " << pci_bandwidth << " GB/s" << std::endl;

    // std::cerr << n << "," << degree << "," << flops << "," << mem_bw << "," << pci_bandwidth_time << "," << pci_bandwidth << "\n";
    
    gpuErrchk(cudaFree(d_image));
    gpuErrchk(cudaFree(d_filter));
    gpuErrchk(cudaFree(d_result));

    delete[] h_image;
    delete[] h_result;
    delete[] h_filter;
    
    return 0;
}