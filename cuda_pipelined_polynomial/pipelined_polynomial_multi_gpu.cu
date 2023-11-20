#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>


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
    std::cerr << seconds << std::endl;
}

int main(int argc, char *argv[]) {
    int64_t n = strtoll(argv[1], NULL, 10);
    int degree = atoi(argv[2]);

    float *h_arr = new float[n];
    float *h_result = new float[n];
    float *h_coeffs = new float[n+1];

    for(int64_t i=0; i<n; i++){
        h_arr[i] = 1;
    }
    for (int i=0; i<=degree; i++){
        h_coeffs[i] = 1;
    }

    int chunk_size;
    if (n <= 500000){
        chunk_size = n/20;
    }
    else if (n <= 2000000){
        chunk_size = n/100;
    }
    else{
        chunk_size = 100000;
    }

    const int num_streams = 20;
    float **d_arr_chunk, *d_coeffs_0, *d_coeffs_1, **d_result_chunk;
    d_arr_chunk = new float*[num_streams];
    d_result_chunk = new float*[num_streams];

    for(int i=0; i<num_streams; i+=2){
        cudaSetDevice(0);
        gpuErrchk(cudaMalloc(&d_arr_chunk[i], chunk_size*sizeof(float)));
        gpuErrchk(cudaMalloc(&d_result_chunk[i], chunk_size*sizeof(float)));
        cudaSetDevice(1);
        gpuErrchk(cudaMalloc(&d_arr_chunk[i+1], chunk_size*sizeof(float)));
        gpuErrchk(cudaMalloc(&d_result_chunk[i+1], chunk_size*sizeof(float)));
    }
    cudaSetDevice(0);
    gpuErrchk(cudaMalloc(&d_coeffs_0, (degree+1)*sizeof(float)));
    gpuErrchk(cudaMemcpy(d_coeffs_0, h_coeffs,  (degree+1)*sizeof(float), cudaMemcpyHostToDevice)); 
    cudaSetDevice(1);
    gpuErrchk(cudaMalloc(&d_coeffs_1, (degree+1)*sizeof(float)));
    gpuErrchk(cudaMemcpy(d_coeffs_1, h_coeffs,  (degree+1)*sizeof(float), cudaMemcpyHostToDevice)); 

    cudaStream_t *stream = new cudaStream_t[num_streams];

    for(int i=0; i<num_streams; i++){
        cudaSetDevice(0);
        gpuErrchk(cudaStreamCreate(&stream[i]));   
        cudaSetDevice(1);
        gpuErrchk(cudaStreamCreate(&stream[i+1]));   
    }

    // Number of chunks
    const int64_t numChunks = n / chunk_size;
    const int threadsPerBlock = 512;
    int blocksPerGrid = (chunk_size + threadsPerBlock - 1) / threadsPerBlock;
    int64_t offset;

    auto start_compute = std::chrono::high_resolution_clock::now();
    for (int64_t i = 0; i < numChunks; i+=num_streams) {

        #pragma unroll
        for (int stream_id=0; stream_id<num_streams; stream_id+=2){
            cudaSetDevice(0);
            offset = (i+stream_id) * chunk_size;
            gpuErrchk(cudaMemcpyAsync(d_arr_chunk[stream_id], &h_arr[offset], chunk_size * sizeof(float), cudaMemcpyHostToDevice, stream[stream_id]));
            compute_poly<<<blocksPerGrid, threadsPerBlock>>>(d_arr_chunk[stream_id], d_coeffs_0, d_result_chunk[stream_id], degree, chunk_size, n);
            gpuErrchk(cudaPeekAtLastError() );
            gpuErrchk(cudaMemcpyAsync(&h_result[offset], d_result_chunk[stream_id], chunk_size * sizeof(float), cudaMemcpyDeviceToHost, stream[stream_id]));

            cudaSetDevice(1);
            offset = (i+stream_id+1) * chunk_size;
            gpuErrchk(cudaMemcpyAsync(d_arr_chunk[stream_id+1], &h_arr[offset], chunk_size * sizeof(float), cudaMemcpyHostToDevice, stream[stream_id+1]));
            compute_poly<<<blocksPerGrid, threadsPerBlock>>>(d_arr_chunk[stream_id+1], d_coeffs_1, d_result_chunk[stream_id+1], degree, chunk_size, n);
            gpuErrchk(cudaPeekAtLastError() );
            gpuErrchk(cudaMemcpyAsync(&h_result[offset], d_result_chunk[stream_id+1], chunk_size * sizeof(float), cudaMemcpyDeviceToHost, stream[stream_id+1]));
        }
        
        #pragma unroll
        for (int stream_id=0; stream_id<num_streams/2; stream_id++){
            cudaSetDevice(0);
            gpuErrchk(cudaStreamSynchronize(stream[stream_id]));
            cudaSetDevice(1);            
            gpuErrchk(cudaStreamSynchronize(stream[stream_id+1]));
        }
    }
    auto end_compute = std::chrono::high_resolution_clock::now();

    for(int64_t i=0; i<n; i++){
        if (h_result[i] != degree+1){
            std::cerr << "Result: " << h_result[i] << " At index: " << i <<  " is wrong" << std::endl;
            // Free allocated memory
            for (int stream_id=0; stream_id<num_streams; stream_id++){
                cudaSetDevice(0);
                gpuErrchk(cudaStreamDestroy(stream[stream_id]));
                cudaSetDevice(1);
                gpuErrchk(cudaStreamDestroy(stream[stream_id+1]));
            }
            for (int stream_id=0; stream_id<num_streams; stream_id++){
                gpuErrchk(cudaFree(d_arr_chunk[stream_id]));
                gpuErrchk(cudaFree(d_result_chunk[stream_id]));
            }
            gpuErrchk(cudaFree(d_coeffs_0));
            gpuErrchk(cudaFree(d_coeffs_1));

            delete[] h_arr;
            delete[] h_result;
            delete[] h_coeffs;
            exit(0);
        }
    }

    get_time_elapsed(start_compute, end_compute, n, degree);
    
    gpuErrchk(cudaFree(d_coeffs_0));
    gpuErrchk(cudaFree(d_coeffs_1));

    delete[] h_arr;
    delete[] h_result;
    delete[] h_coeffs;

    for (int stream_id=0; stream_id<num_streams; stream_id++){
        cudaSetDevice(0);
        gpuErrchk(cudaStreamDestroy(stream[stream_id]));
        cudaSetDevice(1);
        gpuErrchk(cudaStreamDestroy(stream[stream_id+1]));
    }
    for (int stream_id=0; stream_id<num_streams; stream_id++){
        gpuErrchk(cudaFree(d_arr_chunk[stream_id]));
        gpuErrchk(cudaFree(d_result_chunk[stream_id]));
    }
    

    return 0;
}