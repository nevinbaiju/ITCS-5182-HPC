#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <string>

struct Node {
    struct Node * next;
};

const int CACHE_LINE = 2048;

int main(int argc, char* argv[]){
    if (argc < 2){
        std::cout << "Usage: ./write_benchmark <memory size (kb)>\n";
        exit(0);
    }
    int n = std::stoi(argv[1])*(1024/sizeof(struct Node));
    // int allocate_status = posix_memalign((void **)&node_arr, sizeof(Node), n * sizeof(Node));

    int nodes_per_cache = CACHE_LINE/sizeof(struct Node);
    int total_chunks = n/nodes_per_cache;

    
    Node *node_arr;
    int allocate_status = posix_memalign((void **)&node_arr, nodes_per_cache * sizeof(Node), nodes_per_cache * sizeof(Node));

    for(int i=0; i<nodes_per_cache; i++){
        node_arr[i].next = &node_arr[i];
    }
    node_arr[nodes_per_cache-1].next = &node_arr[0];

    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<n; i+=nodes_per_cache){
        node_arr[0].next;
        node_arr[1].next;
        node_arr[2].next;
        node_arr[3].next;
        node_arr[4].next;
        node_arr[5].next;
        node_arr[6].next;
        node_arr[7].next;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> elapsed_seconds = end - start;
    double nano_seconds = elapsed_seconds.count();

    std::cout << "Time taken: " << nano_seconds << " ns" << std::endl;
    std::cout << "Latency: " << nano_seconds / n << " ns"<< std::endl;

    free(node_arr);

    return 0;
}