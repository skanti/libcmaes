#include "CMAES.h"
#include <fstream>
#include <complex>
#include "MathKernels.h"
#include <glob.h>
#include<libgen.h>
#include <iostream>
#include <string>
#include <iomanip>
#include <zconf.h>

struct ToyData1 : public Data {
    void create_synthetic_data() {
        n_data = 50;
        dim = 2;
        x.resize(n_data);
        y.resize(n_data, dim);
        // fill
        MathKernels::logspace(x.data(), -1, 5, n_data);
        dvec params(
                {5.3783e-07, 0.90633, 0.10979, 0.24789, 0.60554, 0.35638, 0.00032026, 0.86119, 0.35704, 0.00014935,
                 1.038});

        std::complex<double> jc(0, 1);
        for (int i = 0; i < n_data; i++) {
            // L + Rs + RQ1 + RQ2 + RQ3
            std::complex<double> r = jc * x[i] * params[0] + params[1]
                                     + params[2] / (1.0 + std::pow<double>(jc * x[i] * params[3], params[4]))
                                     + params[5] / (1.0 + std::pow<double>(jc * x[i] * params[6], params[7]))
                                     + params[8] / (1.0 + std::pow<double>(jc * x[i] * params[9], params[10]));
            y(i, 0) = r.real();
            y(i, 1) = r.imag();
        }
    }
};

struct ToyModel1 : public Model {
    ToyModel1(int n_data_, int dim_) : Model(n_data_, dim_) {
        n_params = 11;
    };

    void evaluate(dvec &x, dvec &params) {
        std::complex<double> jc(0, 1);
        for (int i = 0; i < n_model; i++) {
            // L + Rs + RQ1 + RQ2 + RQ3
            std::complex<double> r = jc * x[i] * params[0] + params[1]
                                     + params[2] / (1.0 + std::pow(jc * x[i] * params[3], params[4]))
                                     + params[5] / (1.0 + std::pow(jc * x[i] * params[6], params[7]))
                                     + params[8] / (1.0 + std::pow(jc * x[i] * params[9], params[10]));
            y(i, 0) = r.real();
            y(i, 1) = r.imag();
        }
    }
};


void transform_scale_shift(double *params, double *params_typical, double *params_tss, int n) {
    for (int i = 0; i < n; i++) {
        params_tss[i] = params[i] * params[i] * params_typical[i];
    }
}

int main(int argc, char *argv[]) {
    std::cout << "***********************************************************" << std::endl;
    //-> data
    ToyData1 toy_data;
    toy_data.create_synthetic_data();
    // <-

    // -> model
    ToyModel1 toy_model(toy_data.n_data, toy_data.dim);
    // <-

    CMAES cmaes(&toy_data, &toy_model);
    dvec x0(toy_model.n_params, 0);
    dvec x_typical({1.0e-07, 1.0, 0.1, 0.1, 1.0, 0.1, 1e-4, 1.0, 0.1, 1e-4, 1.0});
    double sigma0 = 0.5;
    dvec x = cmaes.fmin(x0, sigma0, x_typical, 12, 999, transform_scale_shift);
    return 0;
}