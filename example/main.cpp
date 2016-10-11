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


struct ToyWorld : World {
    int n_data; // <-- number of world points
    int n_dim; // <-- number of dimensions (Impedance has 2 dimensions: z-real and z-imag)
    dvec x; // <-- input 1-D array (frequency)
    dmat y_data, y_model; // <-- output N-D array (z-values)

    // read from .dat file
    void read_data() {
        n_data = 50;
        n_dim = 2;
        x.resize(n_data);
        y_data.reserve_and_resize(n_data, n_dim);
        y_model.reserve_and_resize(n_data, n_dim);

        // fill
        logspace(x.data(), -1, 5, n_data);
        dvec params(
                {5.37e-7, 0.91, 0.11, 0.25, 0.61, 0.36, 3e-4, 0.86119, 0.35, 1e-4, 1.0});

        std::complex<double> jc(0, 1);
        for (int i = 0; i < n_data; i++) {
            // L + Rs + RQ1 + RQ2 + RQ3
            std::complex<double> r = jc * x[i] * params[0] + params[1]
                                     + params[2] / (1.0 + std::pow<double>(jc * x[i] * params[3], params[4]))
                                     + params[5] / (1.0 + std::pow<double>(jc * x[i] * params[6], params[7]))
                                     + params[8] / (1.0 + std::pow<double>(jc * x[i] * params[9], params[10]));
            y_data(i, 0) = r.real();
            y_data(i, 1) = r.imag();
        }
    }

    // actual equivalent circuit units
    void evaluate(dvec &params, int n_params) {
        std::complex<double> jc(0, 1);
        for (int i = 0; i < n_data; i++) {
            // L + Rs + RQ1 + RQ2 + RQ3
            std::complex<double> r = jc * x[i] * params[0] + params[1]
                                     + params[2] / (1.0 + std::pow(jc * x[i] * params[3], params[4]))
                                     + params[5] / (1.0 + std::pow(jc * x[i] * params[6], params[7]))
                                     + params[8] / (1.0 + std::pow(jc * x[i] * params[9], params[10]));
            y_model(i, 0) = r.real();
            y_model(i, 1) = r.imag();
        }
    }

    double cost_func(dvec &params, dvec &params_typical, int n_params) {
        double cost = 0.0;
        for (int i = 0; i < n_dim; i++) {
            cost += least_squares(y_model.memptr(i), y_data.memptr(i), n_data);
        }
        return cost;
    }
};


inline void transform_scale_shift(double *x, double *x_typical, double *x_tss, int n_params) {
    for (int i = 0; i < n_params; i++) {
        x_tss[i] = x[i] * x[i] * x_typical[i];
    }
}

int main(int argc, char *argv[]) {
    std::cout << "***********************************************************" << std::endl;
    //-> world
    ToyWorld toy_world;
    toy_world.read_data();
    // <-

    CMAES::Engine cmaes(&toy_world);
    dvec x_typical({1.0e-07, 1.0, 0.1, 1e-4, 1.0, 0.1, 1e-3, 1.0, 1e-2, 1e-1, 1.0});
    double sigma0 = 1;
    Solution sol = cmaes.fmin(x_typical, sigma0, 10, 99999, transform_scale_shift);
    return 0;
}