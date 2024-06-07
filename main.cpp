#include "chrono.hpp"

#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <fstream>

#include <mpi.h>
#include <omp.h>

using namespace Timings;

const double pi = 3.14159265358979323846;

/*!
    * \brief Function to compute the right-hand side of the Poisson equation
    * \param point Point in the domain
    * \return Value of the right-hand side at the point
*/
std::function<double(std::vector<double>&)> f = [] (std::vector<double>& point) -> double {

    double result = 0;

    result = 8 * pi * pi * sin(2 * pi * point[0]) * cos(2 * pi * point[1]);

    return result;
};

/*!
    * \brief Function to compute the exact solution of the Poisson equation
    * \param point Point in the domain
    * \return Value of the exact solution at the point
*/
std::function<double(std::vector<double>&)> uex = [] (std::vector<double>& point) -> double {

    double result = 0;

    result = sin(2 * pi * point[0]) * cos(2 * pi * point[1]);

    return result;
};

/*!
    * \brief Function to compute the error between two iterations of the numerical solutions
    * \param local_U0 Numerical solution at the previous iteration
    * \param local_U1 Numerical solution at the current iteration
    * \param h Mesh size
    * \param rank Rank of the current MPI process
    * \param size Number of MPI processes
    * \return Error between the exact and numerical solutions
*/
double compute_error (std::vector<std::vector<double>>& U, std::vector<std::vector<double>>& Unew, double h, int start_idx, int local_n) {

    double error = 0.;
    
    #pragma omp parallel for reduction(+:error)
    for (int i = 0; i < local_n; i++) {
        for (std::size_t j = 0; j < U[0].size(); j++) {
            error += h * (Unew[i + start_idx][j] - U[i + start_idx][j]) * (Unew[i + start_idx][j] - U[i + start_idx][j]);
        }
    }

    return std::sqrt(error);
}

/*!
    * \brief Function to compute the L2 norm of the error between the exact and numerical solutions
    * \param h Mesh size
    * \param U Numerical solution
    * \param n Size of the matrix
    * \return L2 norm of the error

*/
double norm2 (double h, std::vector<std::vector<double>>& U, int n) {

    double error = 0.;

    #pragma omp parallel for reduction(+:error)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::vector<double> point = {i * h, j * h};
            error += (uex(point) - U[i][j]) * (uex(point) - U[i][j]);
        }
    }

    return std::sqrt(h * error);
}

/*!
    * \brief Function to write the numerical solution to a VTK file
    * \param U Numerical solution
    * \param n Size of the matrix
    * \param filename Name of the VTK file
*/
void toVTK(const std::vector<std::vector<double>>& U, int n, double h, const std::string& filename) {
    std::ofstream file(filename);

    file << "# vtk DataFile Version 2.0\n";
    file << "Solution data\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << n << " " << n << " 1\n";
    file << "ORIGIN 0 0 0\n";
    file << "SPACING" << h << " " << h << "\n";
    file << "POINT_DATA " << n * n << "\n";
    file << "SCALARS solution double\n";
    file << "LOOKUP_TABLE default\n";

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file << U[i][j] << " ";
        }
        file << "\n";
    }

    file.close();
}

int main (int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int rank, max_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &max_size);

    int n = 0;
    int size = 0;

    std::vector<std::vector<double>> U;
    std::vector<std::vector<double>> Unext;

    bool non_convergence = true;
    double tolerance = 1e-6;
    int max_iter = 1000;
    double h;

    if (rank == 0) {

        std::cout << "Enter the size of the matrix: ";
        std::cin >> n;
        std::cout << "Enter the number of parallel tasks: ";
        std::cin >> size;

        if (size > max_size) {
            std::cerr << "Invalid size" << std::endl;

            MPI_Abort(MPI_COMM_WORLD, 1);
        }   

        U.resize(n);
        for (int i = 0; i < n; i++) {
            U[i].resize(n, 0.);
        }
        Unext.resize(n);
        for (int i = 0; i < n; i++) {
            Unext[i].resize(n, 0.);
        }
    }

    if (rank >= size) {
        MPI_Finalize();
        return 0;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    h = 1.0 / (n - 1);

    int start_idx = 0;

    Chrono time;
    time.start();

    std::vector<int> local_n(size);
    std::vector<int> local_start_idx(size);
    for (int i = 0; i < size; i++) {
        local_n[i] = (n % size > i)? n / size + 1: n / size;
        local_start_idx[i] = start_idx;
        start_idx += local_n[i];
    }

    double err = 0.;

    for (int it = 0; it < max_iter; ++it) {

        if (non_convergence) {

            if (rank == 0) {
                
                if (size > 1) {
                    MPI_Send(U[local_start_idx[rank] + local_n[rank] - 1].data(), n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);

                    #pragma omp parallel for collapse(2)
                    for (int i = 1; i < local_n[rank]; i++) {
                        for (int j = 1; j < n - 1; j++) {
                            std::vector<double> point = {(local_start_idx[rank] + i) * h, j * h};
                            Unext[local_start_idx[rank] + i][j] = 0.25 * (U[local_start_idx[rank] + i - 1][j] + U[local_start_idx[rank] + i + 1][j] + U[local_start_idx[rank] + i][j - 1] + U[local_start_idx[rank] + i][j + 1] + h * h * f(point));
                        }
                    }
                } else {
                    #pragma omp parallel for collapse(2)
                    for (int i = 1; i < local_n[rank] - 1; i++) {
                        for (int j = 1; j < n - 1; j++) {
                            std::vector<double> point = {(local_start_idx[rank] + i) * h, j * h};
                            Unext[local_start_idx[rank] + i][j] = 0.25 * (U[local_start_idx[rank] + i - 1][j] + U[local_start_idx[rank] + i + 1][j] + U[local_start_idx[rank] + i][j - 1] + U[local_start_idx[rank] + i][j + 1] + h * h * f(point));
                        }
                    }
                }

                err = compute_error(U, Unext, h, local_start_idx[rank], local_n[rank]);
                non_convergence = (err > tolerance)? true: false;

                #pragma omp parallel for collapse(2)
                for (int i = 1; i < local_n[rank]; i++) {
                    for (int j = 1; j < n - 1; j++) {
                        U[local_start_idx[rank] + i][j] = Unext[local_start_idx[rank] + i][j];
                    }
                }

            } else if (rank < size - 1) {

                MPI_Recv(U[local_start_idx[rank] - 1].data(), n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(U[local_start_idx[rank] + local_n[rank] - 1].data(), n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
                
                #pragma omp parallel for collapse(2)
                for (int i = 0; i < local_n[rank]; i++) {
                    for (int j = 1; j < n - 1; j++) {
                        std::vector<double> point = {(local_start_idx[rank] + i) * h, j * h};
                        Unext[local_start_idx[rank] + i][j] = 0.25 * (U[local_start_idx[rank] + i - 1][j] + U[local_start_idx[rank] + i + 1][j] + U[local_start_idx[rank] + i][j - 1] + U[local_start_idx[rank] + i][j + 1] + h * h * f(point));
                    }
                }

                err = compute_error(U, Unext, h, local_start_idx[rank], local_n[rank]);
                non_convergence = (err > tolerance)? true: false;

                #pragma omp parallel for collapse(2)
                for (int i = 0; i < local_n[rank]; i++) {
                    for (int j = 1; j < n - 1; j++) {
                        U[local_start_idx[rank] + i][j] = Unext[local_start_idx[rank] + i][j];
                    }
                }

            } else {

                MPI_Recv(U[local_start_idx[rank] - 1].data(), n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                #pragma omp parallel for collapse(2)
                for (int i = 0; i < local_n[rank] - 1; i++) {
                    for (int j = 1; j < n - 1; j++) {
                        std::vector<double> point = {(local_start_idx[rank] + i) * h, j * h};
                        Unext[local_start_idx[rank] + i][j] = 0.25 * (U[local_start_idx[rank] + i - 1][j] + U[local_start_idx[rank] + i + 1][j] + U[local_start_idx[rank] + i][j - 1] + U[local_start_idx[rank] + i][j + 1] + h * h * f(point));
                    }
                }

                err = compute_error(U, Unext, h, local_start_idx[rank], local_n[rank]);
                non_convergence = (err > tolerance)? true: false;

                #pragma omp parallel for collapse(2)
                for (int i = 0; i < local_n[rank]; i++) {
                    for (int j = 1; j < n - 1; j++) {
                        U[local_start_idx[rank] + i][j] = Unext[local_start_idx[rank] + i][j];
                    }
                }
            }

            MPI_Bcast(&non_convergence, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

            time.stop();

        } else {

            if (rank == 0) {
                for (int r = 1; r < size; r++) {
                    for (int j = 0; j < local_n[r]; j++) {
                        MPI_Recv(U[j + local_start_idx[r]].data(), n, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            } else {
                for (int j = 0; j < local_n[rank]; j++) {
                    MPI_Send(Unext[j + local_start_idx[rank]].data(), n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                }
            }

            break;
        }
    }

    if (rank == 0) {
        toVTK(U, n, h, "solution_" + std::to_string(size) + ".vtk");
        std::cout << "L2-norm of the error: " << norm2(h, U, n) << std::endl;
        std::cout << "Execution time: "<<time.wallTime() << " microseconds" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}