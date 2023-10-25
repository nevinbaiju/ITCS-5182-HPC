#!/bin/bash

mkdir results
mkdir plots

rm results/*
rm plots/*

# # PCI-E latency
sbatch --partition=GPU --chdir=`pwd` --time=04:00:00 --ntasks=1 --cpus-per-task=16 --gpus-per-task=1 --job-name=pci_l --mem=10G ./bench.sh 1 1 1 1 1 pci_l 50

# # Bandwidth
sbatch --partition=GPU --chdir=`pwd` --time=04:00:00 --ntasks=1 --cpus-per-task=16 --gpus-per-task=1 --job-name=bw_1 --mem=10G ./bench.sh 5000  5120000 5000 1 1 bw_1 1
sbatch --partition=GPU --chdir=`pwd` --time=04:00:00 --ntasks=1 --cpus-per-task=16 --gpus-per-task=1 --job-name=bw_2 --mem=10G ./bench.sh 5120000 52400000 100000 1 1 bw_2 1

# FLOPS
sbatch --partition=GPU --chdir=`pwd` --time=04:00:00 --ntasks=1 --cpus-per-task=16 --gpus-per-task=1 --job-name=flops --mem=10G ./bench.sh 100000 100000 100000 10000 10000 flops 50
