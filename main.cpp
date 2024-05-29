#include <iostream>
#include <vector>
#include <functional>
#include <cmath>

#include <mpi.h>
#include <omp.h>

//VEDI CASO UNA SOLA RIGA
//GESTISCI RIGA J+1

const double pi = 3.14159265358979323846;

std::function<double(std::vector<double>)> f = [] (std::vector<double> point) -> double {

    double result = 0;

    result = 8 * pi * pi * sin(2 * pi * point[0]) * cos(2 * pi * point[1]);

    return result;
};

double compute_error (std::vector<std::vector<double>> local_U0, std::vector<std::vector<double>> local_U1, double h);

int main (int argv, char* argc[]) {

    MPI_Init(&argv, &argc);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = 0;
    std::vector<std::vector<double>> U;
    int size = 0;

    bool non_convergence = true;
    double tolerance = 1e-6;
    int max_iter = 1000;
    int h;

    if (rank == 0) {

        std::cout << "Enter the size of the matrix: ";
        std::cin >> n;
        U.resize(n);
        for (int i = 0; i < n; i++) {
            U[i].resize(n, 0.);
        }

        std::cout << "Enter the number of parallel tasks: ";
        std::cin >> size;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    h = 1 / (n - 1);

    int start_idx = 0;
    std::vector<int> local_n(size);
    std::vector<int> local_start_idx(size);
    for (int i = 0; i < size; i++) {
        local_n[i] = (n % size > i)? n / size + 1: n / size;
        local_start_idx[i] = start_idx;
        start_idx += local_n[i];
    }
    
    std::vector<std::vector<double>> local_U0;
    std::vector<std::vector<double>> local_U1;
    std::vector<double> previous_row(n, 0.);
    double err = 0.;

    for (int i = 0; i < max_iter; ++i) {

        if (non_convergence) {

            if (rank == 0) {
                
                if (i == 0) {
                    local_U0.resize(local_n[rank]);
                    local_U1.resize(local_n[rank]);

                    for (int j = 0; j < local_n[rank]; j++) {
                        local_U0[j].resize(n, 0.);
                        local_U1[j].resize(n, 0.);
                    }
                }

                for (int j = 1; j < local_n[rank]; j++) {
                    for (int k = 1; k < n - 1; k++) {

                        if (i == 0) {
                            local_U0[j][k] = U[j][k];
                        }

                        local_U1[j][k] = 0.25 * (local_U0[j - 1][k] + local_U0[j + 1][k] + local_U0[j][k - 1] + local_U0[j][k + 1] + h * h * f({static_cast<double>(j * h), static_cast<double>(k * h)}));
                    }
                }

                err = compute_error(local_U0, local_U1, h);
                non_convergence = (err > tolerance)? true: false;

                for (int j = 1; j < local_n[rank]; j++) {
                    for (int k = 1; k < n - 1; k++) {
                        local_U0[j][k] = local_U1[j][k];
                    }
                }

                MPI_Bcast(&non_convergence, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
                MPI_Send(local_U1[local_n[rank] - 1].data(), n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);

            } else if (rank != size - 1) {
                
                if (i == 0) {
                    local_U0.resize(local_n[rank]);
                    local_U1.resize(local_n[rank]);

                    for (int j = 0; j < local_n[rank]; j++) {
                        local_U0[j].resize(n, 0.);
                        local_U1[j].resize(n, 0.);
                    }
                }

                MPI_Recv(previous_row.data(), n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (i == 0) {
                    for (int j = 0; j < local_n[rank]; j++) {
                        for (int k = 1; k < n - 1; k++) {
                            local_U0[j][k] = U[j + local_start_idx[rank]][k];
                        }
                    }
                }

                for (int k = 1; k < n - 1; k++) {
                        local_U1[0][k] = 0.25 * (previous_row[k] + local_U0[1][k] + local_U0[0][k - 1] + local_U0[0][k + 1] + h * h * f({static_cast<double>(local_start_idx[rank] * h), static_cast<double>(k * h)}));;
                }

                for (int j = 0; j < local_n[rank]; j++) {
                    for (int k = 1; k < n - 1; k++) {
                        local_U1[j][k] = 0.25 * (local_U0[j - 1][k] + local_U0[j + 1][k] + local_U0[j][k - 1] + local_U0[j][k + 1] + h * h * f({static_cast<double>((local_start_idx[rank] + j) * h), static_cast<double>(k * h)}));
                    }
                }

                err = compute_error(local_U0, local_U1, h);
                non_convergence = (err > tolerance)? true: false;

                for (int j = 0; j < local_n[rank]; j++) {
                    for (int k = 1; k < n - 1; k++) {
                        local_U0[j][k] = local_U1[j][k];
                    }
                }

                MPI_Bcast(&non_convergence, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
                MPI_Send(local_U1[local_n[rank] - 1].data(), n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);

            } else {
                
                if (i == 0) {
                    local_U0.resize(local_n[rank]);
                    local_U1.resize(local_n[rank]);

                    for (int j = 0; j < local_n[rank]; j++) {
                        local_U0[j].resize(n, 0.);
                        local_U1[j].resize(n, 0.);
                    }
                }

                MPI_Recv(previous_row.data(), n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (i == 0) {
                    for (int j = 0; j < local_n[rank] - 1; j++) {
                        for (int k = 1; k < n - 1; k++) {
                            local_U0[j][k] = U[j + local_start_idx[rank]][k];
                        }
                    }
                }

                for (int k = 1; k < n - 1; k++) {
                        local_U1[0][k] = 0.25 * (previous_row[k] + local_U0[1][k] + local_U0[0][k - 1] + local_U0[0][k + 1] + h * h * f({static_cast<double>(local_start_idx[rank] * h), static_cast<double>(k * h)}));;
                }

                for (int j = 0; j < local_n[rank] - 1; j++) {
                    for (int k = 1; k < n - 1; k++) {
                        local_U1[j][k] = 0.25 * (local_U0[j - 1][k] + local_U0[j + 1][k] + local_U0[j][k - 1] + local_U0[j][k + 1] + h * h * f({static_cast<double>((local_start_idx[rank] + j) * h), static_cast<double>(k * h)}));
                    }
                }

                err = compute_error(local_U0, local_U1, h);
                non_convergence = (err > tolerance)? true: false;

                for (int j = 0; j < local_n[rank] - 1; j++) {
                    for (int k = 1; k < n - 1; k++) {
                        local_U1[j][k] = local_U0[j][k];
                    }
                }

                MPI_Bcast(&non_convergence, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

            }
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
    
    MPI_Finalize();
    return 0;
}

double compute_error (std::vector<std::vector<double>> local_U0, std::vector<std::vector<double>> local_U1, double h) {

    double error = 0.;

    for (std::size_t i = 0; i < local_U0.size(); i++) {
        for (std::size_t j = 0; j < local_U0[i].size(); j++) {
            error += h * (local_U1[i][j] - local_U0[i][j]) * (local_U1[i][j] - local_U0[i][j]);
        }
    }

    return std::sqrt(error);
}