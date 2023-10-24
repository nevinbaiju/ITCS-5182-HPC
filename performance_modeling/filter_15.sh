#!/bin/bash

dimensions=("1024 768" "2048 2048" "8192 8192" "4194304 768" "16777216 768")

for dim in "${dimensions[@]}"; do
    IFS=' ' read -r -a dims <<< "$dim"
    width=${dims[0]}
    height=${dims[1]}

    echo "Running for filter size: 15 Image size $width $height with avx"
    ./performance_modeling_avx "$width" "$height" 15


    # echo "Running for filter size: 15 Image size $width $height with bp"
    # ./performance_modeling_basic_parallel "$width" "$height" 15
done