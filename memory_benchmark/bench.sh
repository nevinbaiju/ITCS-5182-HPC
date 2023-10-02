#!/bin/bash

iterate_and_run() {
  local program=$1

  rm results/results_"$program".txt

  # Iterate from 1KB to 200MB in increments of 512KB
  ./"$program" 1 2>> results/results_"$program".txt
  
  for ((size_kb = 16; size_kb <= 20481; size_kb += 16)); do
    echo "Running for: $size_kb KB"
    ./"$program" "$size_kb" 2>> results/results_"$program".txt
  done

  for ((size_kb = 20480; size_kb <= 204801; size_kb += 512)); do
    echo "Running for: $size_kb KB"
    ./"$program" "$size_kb" 2>> results/results_"$program".txt
  done
}

mkdir results
mkdir plots

# Call the function with the provided program name
iterate_and_run "read"
# iterate_and_run "write"
# iterate_and_run "read_write"
# iterate_and_run "latency"

python plot_results.py