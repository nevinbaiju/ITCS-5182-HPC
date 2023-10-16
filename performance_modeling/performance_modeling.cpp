#include <iostream>
#include <iomanip>
#include <cstring>
#include <immintrin.h>
#include <cmath>
#include <chrono>
#include <omp.h>

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

void print_image(float **image, int width, int height){
    for (int y=0; y<height; y++){
        for (int x=0; x<width; x++){
            std::cout << std::setw(5) << std::setfill(' ') << image[y][x] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void print_register(__m256 __m256_reg){
    float buffer[8];
    _mm256_store_ps(&buffer[0], __m256_reg);
    for(int i=0; i<8; i++){
        std::cout << buffer[i] << ", ";
    }
    std::cout << "\n";
}

void init_image(float **&image, int width, int height, int padding){
    image = new float*[height];
    int pixel_val = 1;
    for (int y = 0; y < height; y++) {
        image[y] = new float[width];
        for (int x = 0; x < width; x++) {

            // Account for padding.
            if ((y<padding)|(x<padding)|(y>=(height-padding))|(x>=(width-padding))){
                image[y][x] = 0;
                continue;   
            }
            else{
                    image[y][x] = pixel_val;
                    pixel_val++;
            }
            // if (x < (width/2)){
            //     image[y][x] = 0;
            // }
            // else{
            //     image[y][x] = 1;
            // }
        }
    }
}

void init_result(float **&image, int width, int height){
    image = new float*[height];
    for (int y = 0; y < height; y++) {
        image[y] = new float[width];
    }
}

void generate_identity_kernel(float **&filter, int filter_size){
    filter = new float*[filter_size];
    for(int i=0; i<filter_size; i++){
        filter[i] = new float[filter_size];
        for(int j=0; j<filter_size; j++){
            filter[i][j] = 0;
        }
    }
    int mid_point = int(filter_size/2);
    filter[mid_point][mid_point] = 1;
}

void free_image(float **image, int m){
    for (int y = 0; y < m; y++) {
            delete[] image[y];
    }
    delete[] image;
}

void print_image_debug(float **image, int start_x, int n, int start_y, int m){
    std::cout << start_x << ": " << n << ","<< start_y << ": " << m << std::endl;
    for (int y=start_y; y<=m; y++){
        for (int x=start_x; x<=n; x++){
            std::cout << std::setw(5) << std::setfill(' ') << image[y][x] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}
void print_filter_debug(float filter[][3], int start_x, int n, int start_y, int m){
    for (int y=start_y; y<=m; y++){
        for (int x=start_x; x<=n; x++){
            std::cout << std::setw(5) << std::setfill(' ') << filter[y][x];
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void convolve(float **image, float **result, float **filter, int start_x, int end_x, int start_y, int end_y, int filter_size){
    int filter_y, filter_x, y, x;
    for(y=start_y; y<end_y; y++){
        for(x=start_x; x<end_x; x++){
            for(filter_y=0; filter_y<filter_size; filter_y++){
                for(filter_x=0; filter_x<filter_size; filter_x++){
                    #ifdef DEBUG
                        std::cout << "result y, x: " << y << ", " << x << "\n";
                        std::cout << "image y, x: " << y+filter_y << ", " << x+filter_x << "\n";
                        std::cout << "Filter y, x: " << filter_y << ", " << filter_x << "\n\n\n";
                    #endif
                    result[y][x] += image[y+filter_y][x+filter_x] * filter[filter_y][filter_x];
                }
            }
        }
    }
}

void convolve_avx(float **image, float **result, float **filter, int start_x, int end_x, int start_y, int end_y, int filter_size){
    int filter_y, filter_x, y, x, filter_pos, filter_pixels=filter_size*filter_size;
    int num_registers = 5, reg_index;
    __m256 filter_register[num_registers], image_register[num_registers], result_register[num_registers], result_buffer;
    for(y=start_y; y<end_y; y++){
        for(x=start_x; x<end_x-1; x+=8){
            result_register[reg_index] = _mm256_loadu_ps(&result[y][x]);
            for(filter_y=0; filter_y<filter_size; filter_y++){
                for(filter_x=0; filter_x<filter_size; filter_x++){
                    reg_index = (filter_y*filter_size + filter_x)%num_registers;

                    filter_register[reg_index] = _mm256_set1_ps(filter[filter_y][filter_x]);
                    image_register[reg_index] = _mm256_loadu_ps(&image[y+filter_y][x+filter_x]);
                    
                    #ifdef PRINT_IMAGE
                    std::cout << filter_y << ":" << filter_x<< "\n";
                    print_register(filter_register[reg_index]);
                    std::cout << y << ":" << x << " to " << y << ":" << x+8 << "\n";
                    print_register(image_register[reg_index]);
                    #endif

                    result_register[reg_index] = _mm256_fmadd_ps(image_register[reg_index], filter_register[reg_index], result_register[reg_index]);
                    
                    #ifdef PRINT_IMAGE
                    std::cout << "Result: \n";
                    print_register(result_register[reg_index]);
                    std::cout << std::endl;
                    #endif

                    
                }
            }
            _mm256_storeu_ps(&result[y][x], result_register[reg_index]);
        }
    }
}

void convolve_blocks(float **image, float **result, float **filter, int width, int height, int filter_size, int block_size){
    int  end_x, end_y;

    for(int y=0; y<height; y+=block_size){
        #pragma omp parallel for
        for(int x=0; x<width; x+=block_size){
            end_y = std::min(y+block_size, height);
            end_x = std::min(x+block_size, width);
            convolve_avx(image, result, filter, x, end_x, y, end_y, filter_size);
            // convolve(image, result, filter, x, end_x, y, end_y, filter_size);
        }
    }
}

int main(int argc, char *argv[]) {
    
    if (argc < 3){
        std::cout << "Usage: ./performance_modeling <width> <height> <kernel_size>\n";
        exit(0);
    }
    int width = std::stoi(argv[1]);
    int height = std::stoi(argv[2]);
    int filter_size = std::stoi(argv[3]);
    int padding = 2*int(filter_size/2);

    int nb_iters = 50;

    float **image, **result, **filter;
    init_image(image, width+padding, height+padding, padding/2);
    init_result(result, width, height);
    generate_identity_kernel(filter, filter_size);
    
    #ifdef PRINT_IMAGE
        print_image(filter, filter_size, filter_size);
        print_image(image, width+padding, height+padding);
    #endif
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<nb_iters; i++)
    {
        // convolve(image, result, filter, 0, width, 0, height, filter_size);
        convolve_blocks(image, result, filter, width, height, filter_size, 64);
    }
    #ifdef PRINT_IMAGE
        print_image(result, width, height);
    #endif
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "mid pixel: " << result[int(height/2)][int(width/2)]/nb_iters << std::endl;  
    print_time_elapsed(start, end, filter_size, width, height, nb_iters);
    
    free_image(image, height+padding);
    free_image(result, height);
    free_image(filter, filter_size);

    return 0;
}
