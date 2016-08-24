#include "CMAES.h"
#include <fstream>
#include <complex>
#include "MathKernels.h"
#include <iostream>
#include <iomanip>
#include "omp.h"


struct ToyWorld : World {
    int n_data; // <- number of world points
    int n_dim; // <- number of dimensions (Impedance has 2 dimensions: z-real and z-imag)
    dvec x; // <- input 1-D array (frequency)
    dmat y_data, y_model; // <- output N-D array (z-values)

    // read from .dat file
    void read_data() {
        n_data = 50;
        n_dim = 2;
        x.resize(n_data);
        y_data.reserve_and_resize(n_data, n_dim);
        y_model.reserve_and_resize(n_data, n_dim);

        // fill
        MathKernels::logspace(x.data(), -1, 5, n_data);
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
            cost += MathKernels::least_squares(y_model.memptr(i), y_data.memptr(i), n_data);
        }
        return cost;
    }
};


inline void transform_scale_shift(double *params, double *params_typical, double *params_tss, int n) {
    for (int i = 0; i < n; i++) {
        params_tss[i] = params[i] * params[i] * params_typical[i];
    }
}

int main(int argc, char *argv[]) {
    std::cout << "***********************************************************" << std::endl;
    //-> world
    ToyWorld toy_world;
    toy_world.read_data();
    // <-

    CMAES cmaes(&toy_world);
    dvec x_typical({1.0e-07, 1.0, 0.1, 1e-4, 1.0, 0.1, 1e-3, 1.0, 1e-2, 1e-1, 1.0});
    double sigma0 = 1;
    dvec x = cmaes.fmin(x_typical, sigma0, 6, 999, transform_scale_shift);
    return 0;
}