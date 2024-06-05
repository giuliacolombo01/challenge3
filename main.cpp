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
double compute_error (std::vector<std::vector<double>>& local_U0, std::vector<std::vector<double>>& local_U1, double h, int rank, int size) {

    double error = 0.;

    if (rank == 0) {
    #pragma omp parallel for reduction(+:error) collapse(2)
        for (std::size_t i = 0; i < local_U1.size(); i++) {
            for (std::size_t j = 0; j < local_U1[1].size(); j++) {
                error += h * (local_U1[i][j] - local_U0[i][j]) * (local_U1[i][j] - local_U0[i][j]);
            }
        }
    } else {
    #pragma omp parallel for reduction(+:error) collapse(2)
        for (std::size_t i = 0; i < local_U1.size() - 1; i++) {
            for (std::size_t j = 0; j < local_U1[1].size(); j++) {
                error += h * (local_U1[i][j] - local_U0[i + 1][j]) * (local_U1[i][j] - local_U0[i + 1][j]);
            }
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
            error += (f(point) - U[i][j]) * (f(point) - U[i][j]);
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

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = 0;
    std::vector<std::vector<double>> U;
    int size = 0;

    bool non_convergence = true;
    double tolerance = 1e-6;
    int max_iter = 1000;
    double h;

    if (rank == 0) {

        std::cout << "Enter the size of the matrix: ";
        std::cin >> n;
        std::cout << "Enter the number of parallel tasks: ";
        std::cin >> size;

        U.resize(n);
        for (int i = 0; i < n; i++) {
            U[i].resize(n, 0.);
        }
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
    
    std::vector<std::vector<double>> local_U0;
    std::vector<std::vector<double>> local_U1;
    double err = 0.;

    for (int i = 0; i < max_iter; ++i) {

        if (non_convergence) {

            if (rank == 0) {
                
                if (i == 0) {
                    local_U0.resize(local_n[rank] + 1);
                    local_U1.resize(local_n[rank]);

                    for (int j = 0; j < local_n[rank]; j++) {
                        local_U0[j].resize(n, 0.);
                        local_U1[j].resize(n, 0.);
                    }
                    local_U0[local_n[rank]].resize(n, 0.);

                    for (int j = 0; j < local_n[rank]; j++) {
                        for (int k = 1; k < n - 1; k++) {
                            local_U0[j][k] = U[j][k];
                        }
                    }
                }

                if (size > 1) {
                    MPI_Recv(local_U0[local_n[rank]].data(), n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

                #pragma omp parallel for collapse(2)
                for (int j = 1; j < local_n[rank]; j++) {
                    for (int k = 1; k < n - 1; k++) {
                        std::vector<double> point = {j * h, k * h};
                        local_U1[j][k] = 0.25 * (local_U0[j - 1][k] + local_U0[j + 1][k] + local_U0[j][k - 1] + local_U0[j][k + 1] + h * h * f(point));
                    }
                }

                err = compute_error(local_U0, local_U1, h, rank, size);
                non_convergence = (err > tolerance)? true: false;

                #pragma omp parallel for collapse(2)
                for (int j = 1; j < local_n[rank]; j++) {
                    for (int k = 1; k < n - 1; k++) {
                        local_U0[j][k] = local_U1[j][k];
                    }
                }

                if (size > 1) {
                    MPI_Send(local_U1[local_n[rank] - 1].data(), n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
                }

            } else if (rank < size - 1) {
                
                if (i == 0) {
                    local_U0.resize(local_n[rank] + 2);
                    local_U1.resize(local_n[rank]);

                    for (int j = 0; j < local_n[rank]; j++) {
                        local_U0[j].resize(n, 0.);
                        local_U1[j].resize(n, 0.);
                    }
                    local_U0[local_n[rank]].resize(n, 0.);
                    local_U0[local_n[rank] + 1].resize(n, 0.);

                    for (int j = 0; j < local_n[rank]; j++) {
                        for (int k = 1; k < n - 1; k++) {
                            local_U0[j + 1][k] = U[j + local_start_idx[rank]][k];
                        }
                    }
                }

                MPI_Recv(local_U0[0].data(), n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(local_U0[local_n[rank] + 1].data(), n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                #pragma omp parallel for collapse(2)
                for (int j = 0; j < local_n[rank]; j++) {
                    for (int k = 1; k < n - 1; k++) {
                        std::vector<double> point = {(local_start_idx[rank] + j) * h, k * h};
                        local_U1[j][k] = 0.25 * (local_U0[j][k] + local_U0[j + 2][k] + local_U0[j + 1][k - 1] + local_U0[j + 1][k + 1] + h * h * f(point));
                    }
                }

                err = compute_error(local_U0, local_U1, h, rank, size);
                non_convergence = (err > tolerance)? true: false;

                #pragma omp parallel for collapse(2)
                for (int j = 0; j < local_n[rank]; j++) {
                    for (int k = 1; k < n - 1; k++) {
                        local_U0[j + 1][k] = local_U1[j][k];
                    }
                }

                MPI_Send(local_U1[0].data(), n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
                MPI_Send(local_U1[local_n[rank] - 1].data(), n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);

            } else {
                
                if (i == 0) {
                    local_U0.resize(local_n[rank] + 1);
                    local_U1.resize(local_n[rank]);

                    for (int j = 0; j < local_n[rank]; j++) {
                        local_U0[j].resize(n, 0.);
                        local_U1[j].resize(n, 0.);
                    }
                    local_U0[local_n[rank]].resize(n, 0.);

                    for (int j = 0; j < local_n[rank] - 1; j++) {
                        for (int k = 1; k < n - 1; k++) {
                            local_U0[j + 1][k] = U[j + local_start_idx[rank]][k];
                    }
                    }
                }

                MPI_Recv(local_U0[0].data(), n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                #pragma omp parallel for collapse(2)
                for (int j = 0; j < local_n[rank] - 1; j++) {
                    for (int k = 1; k < n - 1; k++) {
                        std::vector<double> point = {(local_start_idx[rank] + j) * h, k * h};
                        local_U1[j][k] = 0.25 * (local_U0[j][k] + local_U0[j + 2][k] + local_U0[j + 1][k - 1] + local_U0[j + 1][k + 1] + h * h * f(point));
                    }
                }

                err = compute_error(local_U0, local_U1, h, rank, size);
                non_convergence = (err > tolerance)? true: false;

                #pragma omp parallel for collapse(2)
                for (int j = 0; j < local_n[rank] - 1; j++) {
                    for (int k = 1; k < n - 1; k++) {
                        local_U1[j + 1][k] = local_U0[j][k];
                    }
                }

                MPI_Send(local_U1[0].data(), n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
            }

            MPI_Bcast(&non_convergence, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

            time.stop();

        } else {

            if (rank == 0) {
                for (int j = 0; j < local_n[rank]; j++) {
                    for (int k = 1; k < n - 1; k++) {
                        U[j][k] = local_U1[j][k];
                    }
                }

                for (int r = 1; r < size; r++) {
                    for (int j = 0; j < local_n[rank]; j++) {
                        MPI_Recv(U[j + local_start_idx[r]].data(), n, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            } else {
                for (int j = 0; j < local_n[rank]; j++) {
                    MPI_Send(local_U1[j].data(), n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                }
            }

            break;
        }
    }

    if (rank == 0) {
        std::string filename = "data_" + std::to_string(size) + ".txt";
        std::ofstream result(filename);
        toVTK(U, n, h, "solution_" + std::to_string(size) + ".vtk");
        result << "Matrix size: " << n << "\n";
        result << "Number of processors: " << size << "\n";
        result << "L2 norm of the error: " << norm2(h, U, n) << "\n";
        result << "Execution time: " << time.wallTime() << " microseconds" << "\n";
        result.close();
    }
    
    MPI_Finalize();
    return 0;
}