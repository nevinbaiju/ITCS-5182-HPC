#!/bin/bash

iterate_and_run() {
  local program=$1

  rm results/results_"$program".txt
  
  for ((filter_size = 3; filter_size <= 100; filter_size += 2)); do
    echo "Running for filter size: $filter_size"
    ./"$program" 1024 768 "$filter_size" 2>> results/results_"$program".txt
  done
}

mkdir results
mkdir plots

iterate_and_run "performance_modeling"

# python plot_results.py