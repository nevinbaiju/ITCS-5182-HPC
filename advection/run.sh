#!/bin/bash
touch advection.o
rm advection.o
g++ advection.cpp -o advection.o
./advection.o 103 0.0009
./advection.o 103 0.00009
