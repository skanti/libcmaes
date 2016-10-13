#include "Engine.h"
#include <fstream>
#include <complex>
#include "MathKernels.h"
#include <iostream>
#include <iomanip>
#include "omp.h"

void logspace(double *v, double a, double b, int n) {
    double d = (b - a) / (n - 1);
    for (int i = 0; i < n; i++) {
        v[i] = std::pow(10, d * i);
    }
}

double least_squares(double *v1, double *v2, int n) {
    double sum_least_squares = 0;
    for (int i = 0; i < n; i++) {
        double e = (v1[i] - v2[i]);
        sum_least_squares += e * e;
    }
    return sum_least_squares;
}

void create_synthethic_data(dvec &x, dmat &y_data, int n_data) {

    logspace(x.data(), -1, 5, n_data);
    dvec params0(
            {5e-7, 1.0, 0.1, 0.5, 0.5, 0.1, 1e-4, 0.5, 0.5, 1e-3, 1.0});

    std::complex<double> jc(0, 1);
    for (int i = 0; i < n_data; i++) {
        // L + Rs + RQ1 + RQ2 + RQ3
        std::complex<double> r = jc * x[i] * params0[0] + params0[1]
                                 + params0[2] / (1.0 + std::pow<double>(jc * x[i] * params0[3], params0[4]))
                                 + params0[5] / (1.0 + std::pow<double>(jc * x[i] * params0[6], params0[7]))
                                 + params0[8] / (1.0 + std::pow<double>(jc * x[i] * params0[9], params0[10]));
        y_data(i, 0) = r.real();
        y_data(i, 1) = r.imag();
    }
}

int main(int argc, char *argv[]) {
    std::cout << "***********************************************************" << std::endl;
    //-> create and fill synthetic data
    int n_data = 50;
    int n_dim = 2;
    dvec u;
    dmat z_data, z_model;
    u.resize(n_data);
    z_data.reserve_and_resize(n_data, n_dim);
    z_model.reserve_and_resize(n_data, n_dim);
    create_synthethic_data(u, z_data, n_data);
    // <-

    CMAES::Engine cmaes;
    dvec x0({1.0e-05, 1.0, 0.1, 1e-4, 1.0, 0.1, 1e-3, 1.0, 1e-2, 1e-1, 1.0});

    // -> evaluation function
    auto evaluate = [&](dvec &params, int n_params) {
        std::complex<double> j(0, 1);
        for (int i = 0; i < n_data; i++) {
            // L + Rs + RQ1 + RQ2 + RQ3
            std::complex<double> r = j * u[i] * params[0] + params[1]
                                     + params[2] / (1.0 + std::pow(j * u[i] * params[3], params[4]))
                                     + params[5] / (1.0 + std::pow(j * u[i] * params[6], params[7]))
                                     + params[8] / (1.0 + std::pow(j * u[i] * params[9], params[10]));
            z_model(i, 0) = r.real();
            z_model(i, 1) = r.imag();
        }
    };
    // <-

    // -> cost function
    CMAES::Engine::cost_type cost_func = [&](dvec &x, int n_params) {
        double cost = 0.0;
        evaluate(x, n_params);
        for (int i = 0; i < n_dim; i++) {
            cost += least_squares(z_model.memptr(i), z_data.memptr(i), n_data);
        }
        return cost;
    };
    // <-

    // -> transform-scale-shift function
    CMAES::Engine::tss_type transform_scale_shift = [&](double *x, double *x_tss, int n_params) {
        for (int i = 0; i < n_params; i++) {
            x_tss[i] = std::abs(x[i] * x0[i]);
        }
    };
    // <-

    double sigma0 = 1;
    Solution sol = cmaes.fmin(x0, 11, sigma0, 10, 9999, cost_func, transform_scale_shift);
    return 0;
}