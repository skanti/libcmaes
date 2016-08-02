#include "CMAES.h"
#include <fstream>
#include <complex>
#include "MathKernels.h"
#include <glob.h>
#include<libgen.h>

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

void
write_solution_to_file(std::string filename, dvec &params, int n_params, dvec &x, dmat &y_model, dmat &y_data,
                       int n_data, int dim) {
    std::ofstream file;
    file.open(filename);
    file << n_params << std::endl;
    for (int i = 0; i < n_params; i++)
        file << params[i] << std::endl;

    file << n_data << std::endl;
    for (int i = 0; i < n_data; i++) {
        file << x[i] << std::endl;
    }
    file << dim << std::endl;
    for (int i = 0; i < n_data; i++) {
        file << y_model(i, 0);
        for (int j = 1; j < dim; j++)
            file << " " << y_model(i, j);
        file << std::endl;
    }
    for (int i = 0; i < n_data; i++) {
        file << y_data(i, 0);
        for (int j = 1; j < dim; j++)
            file << " " << y_data(i, j);
        file << std::endl;
    }
    file.close();
}

int main() {
    glob_t glob_result;
    std::string dir_src = "/Users/amon/grive/uni/sofc/700C/700C_filtered";
    std::string dir_target = "/Users/amon/grive/uni/sofc/700C/700C_fitted";
    glob(std::string(dir_src + "/*").c_str(), GLOB_TILDE, NULL, &glob_result);
    for (unsigned int i = 0; i < glob_result.gl_pathc; i++) {
        std::cout << "file " << i << "/" << glob_result.gl_pathc << std::endl;
        std::string filename = glob_result.gl_pathv[i];
        std::string basename1(basename((char *) filename.c_str()));
        std::string rawname = basename1.substr(0, basename1.find_last_of("."));
        //-> data
        ToyData1 toy_data;
        toy_data.read_data_from_file(filename);
        // <-

        // -> model
        ToyModel1 toy_model(toy_data.n_data, toy_data.dim);
        // <-

        CMAES cmaes(&toy_data, &toy_model);
        dvec x0(toy_model.n_params, 0.0);
        dvec x_typical({1.0e-07, 1.0, 0.1, 0.1, 1.0, 0.1, 1e-4, 1.0, 0.1, 1e-4, 1.0});
        double sigma0 = 1;
        dvec x = cmaes.fmin(x0, sigma0, x_typical, 12, 999 + i);
        write_solution_to_file(dir_target + "/" + rawname + ".sol", x, toy_model.n_params, toy_data.x, toy_model.y,
                               toy_data.y, toy_data.n_data, toy_model.dim);
    }
    return 0;
}