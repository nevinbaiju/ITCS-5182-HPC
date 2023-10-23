#!/bin/bash

iterate_and_run() {
  local program=$1
  local compiling=$2

  # Define the dimensions as tuples (width, height)
  dimensions=("1024 768" "2048 2048" "8192 8192" "4194304 768")

  # Iterate over the dimensions
  for dim in "${dimensions[@]}"; do
      # Split the dimension string into width and height
      IFS=' ' read -r -a dims <<< "$dim"
      width=${dims[0]}
      height=${dims[1]}
  
    for ((filter_size = 3; filter_size <= 15; filter_size += 2)); do
      echo "Running for filter size: $filter_size Image size $width $height with $compiling"
      ./"$program" "$width" "$height" "$filter_size" 2>> results/results_"$width"_"$height"_"$program".txt
    done
  done
}

mkdir results
mkdir plots

iterate_and_run "$1" "$2"