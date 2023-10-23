#!/bin/bash

iterate_and_run() {
    local program=$1
    local compiling=$2
    
    for ((filter_size = 3; filter_size <= 15; filter_size += 2)); do
        echo "Running for filter size: $filter_size Image size 16777216 × 768 using $compiled"
        ./"$program" 16777216 768 "$filter_size" 2>> results/results_16777216_768_"$program".txt
    done
    }

mkdir results
mkdir plots

iterate_and_run "$1" "$2"