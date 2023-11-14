#include <stdio.h>
#include<iostream>
#include <cmath> 
#include <cuda_runtime.h>
#include <iomanip>
#include <chrono>
#include <cstring>

#define DEBUG 0

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void print_time_elapsed(std::chrono::time_point<std::chrono::high_resolution_clock> start, std::chrono::time_point<std::chrono::high_resolution_clock> end, 
                        int filter_size, int width, int height, int nb_iters){
    double mega_pixels = (width*height*nb_iters)/1e6;
    double flop = ((filter_size*filter_size) + (filter_size*filter_size - 1))*mega_pixels;                            
    std::chrono::duration<double> elapsed_seconds = end - start;
    double seconds = elapsed_seconds.count();
    double flops = (flop)/(seconds*1e3);
    std::cout << "Time taken: " << seconds  <<  " seconds" << std::endl;
    std::cout << "GFlops: " << flops << std::endl;
    std::cerr << seconds << std::endl;
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

__device__ void print_image_helper(float *image, int width, int height) {
    for (int y=0; y<height; y++){
        for (int x=0; x<width; x++){
            printf("%f,", image[y*width + x]);
        }
        printf("\n");
    }
}

template <int SharedSize>
__global__ void convolve2(float *image, float *res_image, float *filter, int height, int width, int filter_size, int padded_width) {
    __shared__ float shared_image[SharedSize]; // (filter_size*filter_size) * num_threads
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int y = int(idx/width), x = idx%width;
    if (idx < height*width){
        for(int i=0; i<filter_size; i++){
            for(int j=0; j<filter_size; j++){
                shared_image[i*(filter_size*blockDim.x) + (idx*filter_size+j)%(filter_size*blockDim.x)] = image[(y+i)*padded_width + x+j];
            }
        }
    }
    __syncthreads();

    if (idx < height * width) {
        float res = 0;
        
        for (int i = 0; i < filter_size; i++) {
            for (int j = 0; j < filter_size; j++) {
                res += shared_image[i*(filter_size*blockDim.x) + (idx*filter_size+j)%(filter_size*blockDim.x)] * filter[i * filter_size + j];
            }
        }
        
        res_image[idx] = res;
    }
}

void init_image(float *image, int width, int height, int padding){
    long int pixel_val = 1;
    for (long int y = 0; y < height; y++) {
        for (long int x = 0; x < width; x++) {
            // Account for padding.
            if ((y<padding)|(x<padding)|(y>=(height-padding))|(x>=(width-padding))){
                image[y*width + x] = 0;
                continue;   
            }
            else{
                    image[y*width + x] = pixel_val;
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
            std::cout << std::setw(5) << std::setfill(' ') << image[y*width + x] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void validate_result(float *result, long int size){
    for(long int i=0; i<size; i++){
        if (result[i] != i+1){
            std::cerr << "Wrong answer at index: " << i-1 << std::endl;
            #if DEBUG
                return;
            #else
                exit(0);
            #endif
        }
    }
    std::cout << "All pixels checked and verified!\n";
}


int main(int argc, char *argv[]) {
    int height = atoi(argv[1]);
    int width = atoi(argv[2]);
    int filter_size = atoi(argv[3]);

    int padding = 2*int(filter_size/2);

    float* h_image = static_cast<float*>(malloc((height+padding)*(width+padding) * sizeof(float)));
    float* h_result = static_cast<float*>(malloc(height*width*sizeof(float)));
    float* h_filter = new float[filter_size*filter_size];
    
    init_image(h_image, width+padding, height+padding, padding/2);
    generate_identity_kernel(h_filter, filter_size);

    #if DEBUG
        print_image(h_filter, filter_size, filter_size);
        print_image(h_image, width+padding, height+padding);
    #endif

    float *d_image, *d_filter, *d_result;
    gpuErrchk(cudaMalloc(&d_image, (height+padding)*(width+padding)*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_filter, filter_size*filter_size*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_result, height*width*sizeof(float)));
    
    gpuErrchk(cudaMemcpy(d_image, h_image, (height+padding)*(width+padding)*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_filter, h_filter, filter_size*filter_size*sizeof(float), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (height*width + threadsPerBlock - 1) / threadsPerBlock;
    
    auto start_compute = std::chrono::high_resolution_clock::now();
    
    int nb_iters = 1;
    for(int i=0; i<nb_iters; i++){
        convolve<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_result, d_filter, height, width, filter_size, width+padding);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError() );
    }
    
    auto end_compute = std::chrono::high_resolution_clock::now();
    
    gpuErrchk(cudaMemcpy(h_result, d_result, height*width*sizeof(float), cudaMemcpyDeviceToHost));

    #if DEBUG
        print_image(h_result, width, height);
    #endif
    
    validate_result(h_result, height*width);

    print_time_elapsed(start_compute, end_compute, filter_size, width, height, nb_iters);
    
    gpuErrchk(cudaFree(d_image));
    gpuErrchk(cudaFree(d_filter));
    gpuErrchk(cudaFree(d_result));

    delete[] h_image;
    delete[] h_result;
    delete[] h_filter;
    
    return 0;
}