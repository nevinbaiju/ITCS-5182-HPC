#!/bin/bash

mkdir results
mkdir plots

rm results/*
rm plots/*

# # PCI-E latency
for ((i = 0; n <= 50; n += 1)); do
    ./bench.sh 1 1 1 1 1 pci_l
done

# # Bandwidth
./bench.sh 5000  5120000 5000 1 1 bw_1
./bench.sh 5120000 52400000 100000 1 1 bw_2

# FLOPS
for ((i = 0; n <= 50; n += 1)); do
    ./bench.sh 100000 100000 100000 10000 10000 flops
done
