#!/bin/bash

for ((n = $1; n <= $2; n += $3)); do
    for ((deg = $4; deg <= $5; deg += 10)); do
        echo "Running for array size: $n, Degree: $deg, experiment: $5"
        ./cuda_polynomial $n $deg 2>> results/$6.txt
    done
done