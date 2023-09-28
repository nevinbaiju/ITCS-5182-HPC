#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <string>
#include <unordered_set>

int main(int argc, char* argv[]){
    if (argc < 2){
        std::cout << "Usage: ./write_benchmark <memory size (kb)>\n";
        exit(0);
    }
    int n = std::stoi(argv[1])*(1024/sizeof(int));
    int *ll_arr = new int[n];

    int middle = (n-1)/2;
    for(int i=0; i<n; i++){
        if(i == middle){
            ll_arr[i] = 0;
        }
        else if (i > middle){
            ll_arr[i] = (n-1) - i + 1;
        }
        else{
            ll_arr[i] = (n-1) - i;
        }
    }

    int current = 0;
    long int num_accesses = 1e10;
    int nb_iters = 10;

    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<nb_iters; i++){
        for(int i=0; i<n; i++){
            current = ll_arr[current];
            // std::cout << current << std::endl;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Final value: " << current << std::endl;

    std::chrono::duration<double, std::nano> elapsed_seconds = end - start;
    double nano_seconds = elapsed_seconds.count();

    std::cout << "Time taken: " << nano_seconds << " ns" << std::endl;
    std::cout << "Latency: " << nano_seconds / (num_accesses*nb_iters) << " ns"<< std::endl;

    free(ll_arr);

    return 0;
}