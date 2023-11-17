#include <iostream>
#include <cuda_runtime.h>
#include <chrono>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void compute_poly(float *d_arr_chunk, float *d_coeffs, float *d_result_chunk, int degree, int chunk_size, int arr_size){
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    float x, x_tothepowerof, res;
    if (idx < chunk_size){
        x = d_arr_chunk[idx];
        x_tothepowerof = x;
        res = x;
        for(int i=1; i<=degree; i++){
            x_tothepowerof *= x;
            res += d_coeffs[i]*x_tothepowerof;
        }
        d_result_chunk[idx] = res;
    }
}

void get_time_elapsed(std::chrono::time_point<std::chrono::high_resolution_clock> start, 
                        std::chrono::time_point<std::chrono::high_resolution_clock> end, int64_t n, int degree){
    std::chrono::duration<double> elapsed_seconds = end - start;
    double seconds = elapsed_seconds.count();
    int64_t gflop = (((n))*3*(degree + 1))/1e9;
    double flops =  gflop/(seconds);
    std::cout << "FLOPS: " << flops << " Giga FLOPS" << std::endl;
}

int main(int argc, char *argv[]) {
    int64_t n = strtoll(argv[1], NULL, 10);
    int degree = atoi(argv[2]);
    std::cout << "n: " << n << std::endl;

    float *h_arr = new float[n];
    float *h_result = new float[n];
    float *h_coeffs = new float[n+1];

    for(int64_t i=0; i<n; i++){
        h_arr[i] = 1;
    }
    for (int i=0; i<=degree; i++){
        h_coeffs[i] = 1;
    }

    int chunk_size = n/10;

    float *d_arr_chunk, *d_coeffs, *d_result_chunk;

    gpuErrchk(cudaMalloc(&d_arr_chunk, chunk_size*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_result_chunk, chunk_size*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_coeffs, (degree+1)*sizeof(float)));
    gpuErrchk(cudaMemcpy(d_coeffs, h_coeffs,  (degree+1)*sizeof(float), cudaMemcpyHostToDevice)); 

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    // Number of chunks
    const int64_t numChunks = n / chunk_size;
    const int threadsPerBlock = 512;
    int blocksPerGrid = (chunk_size + threadsPerBlock - 1) / threadsPerBlock;
    int64_t offset;
    auto start_compute = std::chrono::high_resolution_clock::now();
    for (int64_t i = 0; i < numChunks; ++i) {
        offset = i * chunk_size;
        
        gpuErrchk(cudaMemcpyAsync(d_arr_chunk, &h_arr[offset], chunk_size * sizeof(float), cudaMemcpyHostToDevice, stream));
        compute_poly<<<blocksPerGrid, threadsPerBlock>>>(d_arr_chunk, d_coeffs, d_result_chunk, degree, chunk_size, n);
        gpuErrchk(cudaPeekAtLastError() );
        gpuErrchk(cudaMemcpyAsync(&h_result[offset], d_result_chunk, chunk_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
        
        gpuErrchk(cudaStreamSynchronize(stream));
    }
    auto end_compute = std::chrono::high_resolution_clock::now();

    for(int64_t i=0; i<n; i++){
        if (h_result[i] != degree+1){
            std::cout << "Result: " << h_result[i] << " At index: " << i <<  " is wrong" << std::endl;
            // Free allocated memory
            gpuErrchk(cudaFree(d_arr_chunk));
            gpuErrchk(cudaFree(d_result_chunk));
            gpuErrchk(cudaFree(d_coeffs));

            delete[] h_arr;
            delete[] h_result;
            delete[] h_coeffs;
            exit(0);
        }
    }

    get_time_elapsed(start_compute, end_compute, n, degree);

    // Free allocated memory
    gpuErrchk(cudaFree(d_arr_chunk));
    gpuErrchk(cudaFree(d_result_chunk));
    gpuErrchk(cudaFree(d_coeffs));

    delete[] h_arr;
    delete[] h_result;
    delete[] h_coeffs;

    gpuErrchk(cudaStreamDestroy(stream));

    return 0;
}