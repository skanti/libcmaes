#include "CMAES.h"
#include <string>
#include <fstream>
#include <iostream>

struct ToyData1 : public Data {
    void populate() {
        n_data = 50;
        dim = 2;
        x.resize(n_data);
        y.resize(dim);
        for (int i = 0; i < dim; i++)
            y[i].resize(n_data);
        // fill
        x = arma::logspace(-1, 5, n_data);
        dvec params(
                {5.3783e-07, 0.90633, 0.10979, 0.24789, 0.60554, 0.35638, 0.00032026, 0.86119, 0.35704, 0.00014935,
                 1.038});

        std::complex<double> jc(0, 1);
        for (int i = 0; i < n_data; i++) {
            // L + Rs + RQ1 + RQ2 + RQ3
            std::complex<double> r = jc * x[i] * params[0] + params[1]
                                     + params[2] / (1.0 + std::pow(jc * x[i] * params[3], params[4]))
                                     + params[5] / (1.0 + std::pow(jc * x[i] * params[6], params[7]))
                                     + params[8] / (1.0 + std::pow(jc * x[i] * params[9], params[10]));
            y[0][i] = r.real();
            y[1][i] = r.imag();
        }
    }

};

struct ToyModel1 : public Model {
    ToyModel1(int n_data_, int dim_) : Model(n_data_, dim_) {
        n_params = 11;
    };

    void evaluate(dvec &x, dvec &params) {
        std::complex<double> jc = std::complex<double>(0, 1);
        for (int i = 0; i < n_model; i++) {
            // L + Rs + RQ1 + RQ2 + RQ3
            std::complex<double> r = jc * x[i] * params[0] + params[1]
                                     + params[2] / (1.0 + std::pow(jc * x[i] * params[3], params[4]))
                                     + params[5] / (1.0 + std::pow(jc * x[i] * params[6], params[7]))
                                     + params[8] / (1.0 + std::pow(jc * x[i] * params[9], params[10]));
            y[0][i] = r.real();
            y[1][i] = r.imag();
        }
    }

    void write_model(std::string filename, dvec &x, dvec &params) {
        evaluate(x, params);
        std::ofstream file;
        file.open(filename);
        for (int i = 0; i < n_model; i++) {
            file << y[0][i] << " " << y[1][i];
            file << std::endl;
        }
        file.close();
    }
};

int main() {
    //-> data
    ToyData1 toy_data;
    toy_data.populate();
    // <-

    // -> model
    ToyModel1 toy_model(toy_data.n_data, toy_data.dim);
    // <-

    CMAES cmaes(&toy_data, &toy_model);
    dvec x0(toy_model.n_params, arma::fill::ones);
    dvec x_typical({1.0e-07, 1.0, 0.1, 0.1, 1.0, 0.1, 1e-4, 1.0, 0.1, 1e-4, 1.0});
    double sigma0 = 2;
    cmaes.fmin(x0, sigma0, x_typical, 7, 999);

    // plotting
    /*
    dvec params(std::vector<double>(
    { 5.3783e-07, 0.90633, 0.10979, 0.24789, 0.60554, 0.35638, 0.00032026, 0.86119, 0.35704, 0.00014935, 1.038 }));
    toy_model.plot_model(toy_data.x, params);
     */
    //
    return 0;
}