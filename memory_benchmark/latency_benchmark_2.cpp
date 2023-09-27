#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <string>

struct node {
  int val;
  struct node *next;
};

int main(int argc, char* argv[]){
    if (argc < 2){
        std::cout << "Usage: ./write_benchmark <memory size (kb)>\n";
        exit(0);
    }
    int n = std::stoi(argv[1])*(1024/sizeof(node));

    struct node *head;
    struct node *current;
    struct node *new_node;
    head = malloc(sizeof(struct node));
    head->val = 0;
    current = head;
    for(int i=1; i<n; i++){
        new_node = malloc(sizeof(struct node));
        new_node->val = i;
        current->next = new_node;
        current = current->next;
    }
    current->next = NULL;
    current = head;

    auto start = std::chrono::high_resolution_clock::now();
    while(current != NULL){
        current = current->next;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> elapsed_seconds = end - start;
    double nano_seconds = elapsed_seconds.count();

    std::cout << "Time taken: " << nano_seconds << " ns" << std::endl;
    std::cout << "Latency: " << nano_seconds / n << " ns"<< std::endl;

    current = head;
    struct node prev;
    while(current != NULL){
        current = current->next;
    }

    return 0;
}