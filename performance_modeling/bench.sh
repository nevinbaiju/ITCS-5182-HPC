#!/bin/bash

iterate_and_run() {
  local program=$1

  # Define the dimensions as tuples (width, height)
  dimensions=("1024 768" "2048 768" "8192 768" "4194304 768")

  # Iterate over the dimensions
  rm results/*
  for dim in "${dimensions[@]}"; do
      # Split the dimension string into width and height
      IFS=' ' read -r -a dims <<< "$dim"
      width=${dims[0]}
      height=${dims[1]}
  
    for ((filter_size = 3; filter_size <= 13; filter_size += 2)); do
      echo "Running for filter size: $filter_size Image size $width $height"
      ./"$program" "$width" "$height" "$filter_size" 2>> results/results_"$width"_"$height"_"$program".txt
    done
  done
}

mkdir results
mkdir plots

iterate_and_run "performance_modeling"