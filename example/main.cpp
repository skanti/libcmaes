#include "CMAES.h"
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <complex>
#include "Model.h"
#include "SolverPool.h"
#include "Types.h"
#include "Data.h"
#include <string>

struct ToyData1 : public Data {
    void create_synthetic_data() {
        n_data = 50;
        dim = 2;
        x.resize(n_data);
        y.resize(n_data, dim);
        // fill
        SolverPool::logspace(x.data(), -1, 5, n_data);
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
            y(i, 0) = r.real();
            y(i, 1) = r.imag();
        }
    }

    void read_data_from_file(std::string filename) {
        dim = 2;
        // -> read
        std::string line;
        std::ifstream myfile(filename);
        if (myfile.is_open()) {
            std::getline(myfile, line);
            n_data = std::stoi(line);
            x.resize(n_data);
            y.resize(n_data, dim);
            for (int i = 0; i < n_data; i++) {
                std::getline(myfile, line);
                x[i] = std::stod(line);
            }
            for (int i = 0; i < n_data; i++) {
                std::getline(myfile, line);
                y(i, 0) = std::stod(line);
            }
            for (int i = 0; i < n_data; i++) {
                std::getline(myfile, line);
                y(i, 1) = std::stod(line);
            }
            myfile.close();
        }
        // <-
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

    void write_model(std::string filename, dvec &x, dvec &params) {
        evaluate(x, params);
        std::ofstream file;
        file.open(filename);
        for (int i = 0; i < n_model; i++) {
            file << y(i, 0) << " " << y(i, 1);
            file << std::endl;
        }
        file.close();
    }
};

int main() {
    //-> data
    ToyData1 toy_data;
    toy_data.read_data_from_file("/Users/amon/grive/uni/sofc/code/evaluate_data/z106.dat");
    // <-

    // -> model
    ToyModel1 toy_model(toy_data.n_data, toy_data.dim);
    // <-

    CMAES cmaes(&toy_data, &toy_model);
    dvec x0(toy_model.n_params, 1.0);
    dvec x_typical({1.0e-07, 1.0, 0.1, 0.1, 1.0, 0.1, 1e-4, 1.0, 0.1, 1e-4, 1.0});
    double sigma0 = 1;
    cmaes.fmin(x0, sigma0, x_typical, 10, 999);


    return 0;
}