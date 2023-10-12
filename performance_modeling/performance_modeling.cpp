#include <iostream>
#include <iomanip>
#include <cstring>
#include <immintrin.h>
#include <cmath>
#include <chrono>

#define DEBUG 0

void print_time_elapsed(std::chrono::time_point<std::chrono::high_resolution_clock> start, std::chrono::time_point<std::chrono::high_resolution_clock> end, int num_pixels){
    auto elapsed_seconds = end-start;
    double seconds = elapsed_seconds.count();
    float inferences_per_second = (num_pixels)/(seconds);
    // Print the data
    std::cout << "Pixels per second: " << inferences_per_second << std::endl;
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

void init_image(float **&image, int width, int height, int padding){
    image = new float*[height];
    for (int y = 0; y < height; y++) {
        image[y] = new float[width];
        for (int x = 0; x < width; x++) {

            // Account for padding.
            if ((y<padding)|(x<padding)|(y>=(height-padding))|(x>=(width-padding))){
                image[y][x] = 0;
                continue;   
            }
            if (x < (width/2)){
                image[y][x] = 0;
            }
            else{
                image[y][x] = 1;
            }
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

void convolve(float **image, float **result, float **filter, int width, int height, int filter_size){
    int filter_y, filter_x, y, x;
    for(y=0; y<height; y++){
        for(x=0; x<width; x++){
            for(filter_y=0; filter_y<filter_size; filter_y++){
                for(filter_x=0; filter_x<filter_size; filter_x++){
                    #if DEBUG
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

void convolve_avx(float **image, float **result, float filter[][3], int n, int m){
    int left_x, right_x, top_y, bot_y, conv_index;
    __m256 filter_register = _mm256_setr_ps( filter[0][0], filter[0][1], filter[0][2], 
                                            filter[1][0], filter[1][1], filter[1][2], 
                                            filter[2][0], filter[2][1]);                                            
    float filter_last_val = filter[2][2];

    __m256 image_register, result_register;
    float image_last_val;
    float result_buffer[8];

    for(int y=0; y<m; y++){
        for(int x=0; x<n; x++){

            #if DEBUG
                print_image_debug(image, left_x, right_x, top_y, bot_y);
                print_filter_debug(filter, 0, 2, 0, 2);
            #endif
            image_register = _mm256_setr_ps( image[y][x], image[y][x+1], image[y][x+2], 
                                             image[y+1][x], image[y+1][x+1], image[y+1][x+2], 
                                             image[y+2][x], image[y+2][x+1]);

            result_register = _mm256_mul_ps(filter_register, image_register);                       

            result[y][x] = std::fma(filter_last_val, image[y+2][x+2], hsum_float_avx(result_register));

            #if DEBUG
                std::cout << y << "," << x << "->" << result[y][x] << "\n\n";
            #endif
        }
    }
}

int main(int argc, char *argv[]) {
    
    if (argc < 3){
        std::cout << "Usage: ./performance_modeling <width> <height>\n";
        exit(0);
    }
    int width = std::stoi(argv[1]);
    int height = std::stoi(argv[2]);
    int filter_size = std::stoi(argv[3]);
    int padding = 2*int(filter_size/2);

    float **image, **result, **filter;
    init_image(image, width+padding, height+padding, padding/2);
    init_result(result, width, height);
    generate_identity_kernel(filter, filter_size);
    
    print_image(filter, filter_size, filter_size);
    print_image(image, width+padding, height+padding);
    auto start = std::chrono::high_resolution_clock::now();
    convolve(image, result, filter, width, height, filter_size);
    print_image(result, width, height);
    auto end = std::chrono::high_resolution_clock::now();
    print_time_elapsed(start, end, width*height);


    // start = std::chrono::high_resolution_clock::now();
    // convolve_avx(image, result, sobelVertical, width, height);
    // end = std::chrono::high_resolution_clock::now();
    // print_time_elapsed(start, end, width*height);
    // // print_image(result, width, height);
    
    free_image(image, height+2);
    free_image(result, height);

    return 0;
}
