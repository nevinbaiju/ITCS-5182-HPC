#!/bin/bash

lower_limit=1000000000
upper_limit=10000000000

n=$lower_limit
increment=$lower_limit
next_increment=$((increment * 10))

while [ $n -le $upper_limit ]; do
    if [ $n -eq $next_increment ]; then
        increment=$next_increment
        next_increment=$((increment * 10))
    fi
    ./pipelined_polynomial $n 1 2>> results/flops.txt
    for ((deg = 10; deg <= 100; deg += 10)); do
        echo "Running for size: $n degree: $deg"
        ./pipelined_polynomial $n $deg 2>> results/flops.txt
    done
    n=$((n + increment))
done