    #include "CMAES.h"
#include <string>
#include <fstream>
#include <iostream>

struct ToyData1 : public Data {
    void populate() {
        n = 50;
        dim = 1;
        x.resize(n);
        y.resize(dim);
        y[0].resize(n);
        // fill
        for (int i = 0; i < n; i++)
            x[i] = -4.0 + 8.0 * i / (n - 1.0);
        std::vector<double> params = {{90.0, 0.5, 1.5, -3.5, 1.2, 0.9}};

        for (int i = 0; i < n; i++) {
            double x2 = x[i] * x[i];
            y[0][i] = params[0] * std::exp(-params[1] * x2) + params[2] * x2 * x2 -
                      params[3] / (params[4] + params[5] * x2);
        }
    }

};

struct ToyModel1 : public Model {
    ToyModel1(int n_data_, int dim_, int n_params_) : Model(n_data_, dim_, n_params_) { };

    void evaluate(dvec &x, dvec &params) {
        for (int i = 0; i < n_data; i++) {
            double x2 = x[i] * x[i];
            y[0][i] = params[0] * std::exp(-params[1] * x2) + params[2] * x2 * x2 -
                      params[3] / (params[4] + params[5] * x2);
        }
    }

    void write_data(std::string filename, dvec &x, dvec &params) {
        evaluate(x, params);
        std::ofstream file;
        file.open(filename);
        for (int i = 0; i < n_data; i++) {
            file << x[i];
            for (int j = 0; j < dim; j++)
                file << " " << y[j][i];
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
    ToyModel1 toy_model(toy_data.n, toy_data.dim, 6);
    // <-

    CMAES cmaes(&toy_data, &toy_model);
    dvec x0(6, arma::fill::zeros);
    double sigma0 = 20.0;
    cmaes.fmin(x0, sigma0, 0);
    return 0;
}