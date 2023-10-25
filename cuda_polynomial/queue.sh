#!/bin/bash

mkdir results
mkdir plots

rm results/*
rm plots/*

# # PCI-E latency
./bench.sh 1 1 1 1 1 pci_l 50

# # Bandwidth
./bench.sh 5000  5120000 5000 1 1 bw_1 1
./bench.sh 5120000 52400000 100000 1 1 bw_2 1

# FLOPS
./bench.sh 100000 100000 100000 10000 10000 flops 50
