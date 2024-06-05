#!/bin/bash

# Number of processors to test
PROCS=(1 2 4)

# Grid size
N=64

# Output data folder
DATA_FOLDER="data"
mkdir -p $DATA_FOLDER  # Create the data folder if it doesn't exist

# Compile the program
mpicxx -fopenmp -o hybrid_parallel_program main.cpp

# Run tests with different number of processors
for PROC in "${PROCS[@]}"
do
    mpirun -np $PROC ./main $N $PROC
    echo "Test with $PROC processors completed."
done