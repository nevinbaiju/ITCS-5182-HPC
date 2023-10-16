#!/bin/bash
#!/bin/bash

iterate_and_run() {
    local program=$1
    
    for ((filter_size = 3; filter_size <= 13; filter_size += 2)); do
        echo "Running for filter size: $filter_size Image size 16777216 × 768"
        ./"$program" 16777216 768 "$filter_size" 2>> results/results_16777216_768_"$program".txt
    done
    }

mkdir results
mkdir plots

iterate_and_run "performance_modeling"