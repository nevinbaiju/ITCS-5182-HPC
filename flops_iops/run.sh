#!/bin/bash

g++ flops_intrinsic.cpp compute_functions.cpp \
-march=native -fopenmp -mavx2 -march=native -mtune=haswell -O0 -o flops

for i in {1..1000}; do
    echo "Running iteration $i"
    ./flops source 2>> results_O0
done

g++ flops_intrinsic.cpp compute_functions.cpp \
-march=native -fopenmp -mavx2 -march=native -mtune=haswell -O1 -o flops

for i in {1..1000}; do
    echo "Running iteration $i"
    ./flops source 2>> results_O1
done

g++ flops_intrinsic.cpp compute_functions.cpp \
-march=native -fopenmp -mavx2 -march=native -mtune=haswell -O2 -o flops

for i in {1..1000}; do
    echo "Running iteration $i"
    ./flops source 2>> results_O2
done

g++ flops_intrinsic.cpp compute_functions.cpp \
-march=native -fopenmp -mavx2 -march=native -mtune=haswell -O3 -o flops

for i in {1..1000}; do
    echo "Running iteration $i"
    ./flops source 2>> results_O3
done

python plot_results.py