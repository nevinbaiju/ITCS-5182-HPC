#!/bin/bash

for ((filter_size = 3; filter_size <= 15; filter_size += 2)); do
    echo "Running for filter size: $filter_size Image size 16777216 768"
    ./cuda_convolution 16777216 768 "$filter_size" 2>> results/results_16777216_768
done