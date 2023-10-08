#include <iostream>
#include <iomanip>
#include <cstring> // For memset

#define DEBUG 0

void print_image(float **image, int width, int height){
    for (int y=0; y<height; y++){
        for (int x=0; x<width; x++){
            std::cout << std::setw(5) << std::setfill(' ') << image[y][x] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void init_image(float **&image, int width, int height){
    image = new float*[height];
    for (int y = 0; y < height; y++) {
        image[y] = new float[width];
        for (int x = 0; x < width; x++) {

            // Account for padding.
            if ((y==0)|(x==0)|(y==height-1)|(x==width-1)){
                image[y][x] = 0;
                continue;   
            }
            if (x < width/2){
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

void convolve(float **image, float **result, float filter[][3], int n, int m){
    int left_x, right_x, top_y, bot_y, conv_index;
    for(int y=0; y<m; y++){
        for(int x=0; x<n; x++){
            left_x = x;
            right_x = x+2;
            top_y = y;
            bot_y = y+2;

            #if DEBUG
                print_image_debug(image, left_x, right_x, top_y, bot_y);
                print_filter_debug(filter, 0, 2, 0, 2);
            #endif

            result[y][x] += image[y][x] *  filter[0][0];
            result[y][x] += image[y][x+1] *  filter[0][1];
            result[y][x] += image[y][x+2] *  filter[0][2];

            result[y][x] += image[y+1][x] *  filter[1][0];
            result[y][x] += image[y+1][x+1] *  filter[1][1];
            result[y][x] += image[y+1][x+2] *  filter[1][2];

            result[y][x] += image[y+2][x] *  filter[2][0];
            result[y][x] += image[y+2][x+1] *  filter[2][1];
            result[y][x] += image[y+2][x+2] *  filter[2][2];

            #if DEBUG
                std::cout << y << "," << x << "->" << result[y][x] << "\n\n";
            #endif
        }
    }
}

int main(int argc, char *argv[]) {
    
    if (argc < 2){
        std::cout << "Usage: ./performance_modeling <width> <height>\n";
        exit(0);
    }
    int width = std::stoi(argv[1]);
    int height = std::stoi(argv[2]);

    float **image, **result;
    init_image(image, width+2, height+2);
    init_result(result, width, height);
    float sobelVertical[3][3] = {
        {-1.0, 0.0, 1.0},
        {-2.0, 0.0, 2.0},
        {-1.0, 0.0, 1.0}
    };
    
    // print_image(image, width+2, height+2);
    // print_image(result, width, height);
    convolve(image, result, sobelVertical, width, height);
    // print_image(result, width, height);
    
    free_image(image, height+2);
    free_image(result, height);


    return 0;
}