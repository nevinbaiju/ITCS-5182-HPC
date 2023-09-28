#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <string>
#include <unordered_set>

void iterate_ll(int *ll, int n){
    for(int i=0; i<n; i++){
        std::cout << i << ":" << ll[i] << std::endl;
    }
}

int main(int argc, char* argv[]){
    if (argc < 2){
        std::cout << "Usage: ./write_benchmark <memory size (kb)>\n";
        exit(0);
    }
    int n = std::stoi(argv[1])*(1024/sizeof(int));
    int *ll_arr = new int[n];

    for(int i=0; i<n; i++){
        ll_arr[i] = 0;
    }
    int current = 0;
    int assigned = 1;
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    int next_index_candidate = std::rand() % (n);
    while(next_index_candidate == 0){
        next_index_candidate = std::rand() % (n);
    }
    
    ll_arr[current] = next_index_candidate;
    current = ll_arr[current];

    while(assigned < n-1){
        next_index_candidate = std::rand() % (n);
        while((ll_arr[next_index_candidate] != 0) | (next_index_candidate == current) | (current == ll_arr[next_index_candidate])){
            next_index_candidate = std::rand() % (n);
        }
        ll_arr[current] = next_index_candidate;
        current = ll_arr[current];
        assigned++;
    }
    std::cout << "Finished creating Linked List" << std::endl;


    current = 0;
    long int num_accesses = 1e6;
    int nb_iters = 10;

    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<nb_iters; i++){
        for(int i=0; i<num_accesses; i++){
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