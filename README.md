
# Matrix–free parallel solver for the Laplace equation

The code solves the Laplace equation, modelling the heat diffusion over a square domain with a prescribed temperature on the whole boundary. To do that it uses Jacobi iteration method.
The solution of the equation U is represented as a sparse matrix n x n containig the evaluations of the discrete solution in some nodes of a Cartesian grid.

## First code - description

`main.cpp` solves the iterative algorithm by using MPI and OpenMP: each rank updates a certain number of rows of U using the prescribed rule, then it sends the result to rank 0 which exports the solution in vtk format. The size of the matrix and the number of processors to be used are given in input by the user.

## First code - how to run

To run and compile this code there's a Makefile, so it sufficies to do (in this order):
```bash
   make
```
```bash
   ./main
```

## Second code - description

`main1.cpp` solves the iterative algorithm in the same way of the previous one, but now there's no input by the user.

`scalability.sh` compiles and executes the program with a different number of processors (PROCS) on a grid of dimesion N. These parameters can be changed in the code by the user.

`plot.py` creates two graphs to see the execution time and the L2-norm in function of the number of processors.

## Second code - how to run

To compile this code, to run all the test and to create the two plot you have to do (in this order):
```bash
   mpicxx -fopenmp -o main1 main1.cpp
```
This compiles the code and creates the executable.

```bash
  mkdir data
```
This creates the directory for the data.

```bash
  ./scalability.sh
```
```bash
  python plot.py
```
It may be necessary to install the library matplotlib before running this last command; in this case do
```bash
  pip install --user matplotlib
```