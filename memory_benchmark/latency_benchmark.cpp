#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <string>

struct Node {
    struct Node * next;
};

int main(int argc, char* argv[]){
    if (argc < 2){
        std::cout << "Usage: ./write_benchmark <memory size (kb)>\n";
        exit(0);
    }
    Node *node_arr;
    int n = std::stoi(argv[1])*128;
    int allocate_status = posix_memalign((void **)&node_arr, sizeof(Node), n * sizeof(Node));

    for(int i=0; i<n; i++){
        node_arr[i].next = &node_arr[i];
    }

    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<n; i++){
        node_arr[i].next;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> elapsed_seconds = end - start;
    double nano_seconds = elapsed_seconds.count();

    std::cout << "Time taken: " << nano_seconds << " ns" << std::endl;
    std::cout << "Latency: " << nano_seconds / n << " ns"<< std::endl;

    free(node_arr);

    return 0;
}