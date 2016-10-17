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

void create_synthethic_data(dvec &x, dvec &x00, dmat &y_data, int n_data) {

    logspace(x.data(), -1, 5, n_data);

    std::complex<double> jc(0, 1);
    for (int i = 0; i < n_data; i++) {
        // L + Rs + RQ1 + RQ2 + RQ3
        std::complex<double> r = jc * x[i] * x00[0] + x00[1]
                                 + x00[2] / (1.0 + std::pow<double>(jc * x[i] * x00[3], x00[4]))
                                 + x00[5] / (1.0 + std::pow<double>(jc * x[i] * x00[6], x00[7]))
                                 + x00[8] / (1.0 + std::pow<double>(jc * x[i] * x00[9], x00[10]));
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
    dvec x00({4.69624e-6, 3.0407, 1.01319, 0.00820486, 0.369616, 1.0105, 0.00187691, 0.681281, 0.875321, 0.0113654,
              0.873393});
    z_data.reserve_and_resize(n_data, n_dim);
    z_model.reserve_and_resize(n_data, n_dim);
    create_synthethic_data(u, x00, z_data, n_data);
    // <-

    CMAES::Engine cmaes;
    dvec x0(11);
    for (int i = 0; i < 11; i++) {
        double h1 = 2;
        x0[i] = x00[i] * (i % 2 == 0 ? h1 : 1.0 / h1);
    }
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


    double sigma0 = 5;
    int seed = 42;
    int n_params = 11;
    int n_restarts = 10;
    Solution sol = cmaes.fmin(x0, n_params, sigma0, n_restarts, seed, cost_func, transform_scale_shift);
    std::cout << "f_min: " << sol.f << " i_func_eval: " << sol.i_func_evaluations << std::endl;
    return 0;
}