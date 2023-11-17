#!/bin/bash

mkdir results
mkdir plots

rm results/*
rm plots/*

sbatch --partition=GPU --chdir=`pwd` --time=04:00:00 --ntasks=1 --cpus-per-task=16 --gpus-per-task=1 --job-name=bw_1 --mem=32G ./bench.sh
sbatch --partition=GPU --chdir=`pwd` --time=04:00:00 --ntasks=1 --cpus-per-task=16 --gpus-per-task=1 --job-name=bw_2 --mem=100G ./bench_last.sh