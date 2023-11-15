#!/bin/bash

rm results/*
rm plots/*

mkdir results
mkdir plots

# sbatch --partition=GPU --chdir=`pwd` --time=04:00:00 --ntasks=1 --cpus-per-task=16 --gpus-per-task=1 --job-name=pci_l --mem=100G ./bench_last.sh
sbatch --partition=GPU --chdir=`pwd` --time=04:00:00 --ntasks=1 --cpus-per-task=16 --gpus-per-task=1 --job-name=bw_1 --mem=64G ./bench.sh
