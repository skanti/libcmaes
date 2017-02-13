#include "Engine.h"
#include <fstream>
#include <iostream>
#include "omp.h"

#define N_DIMENSION 3


void create_synthethic_data(dvec &u, dvec &y_data, int n_data) {

    std::iota(&u[0], &u[0] + n_data, 0.0);
    dvec params0(N_DIMENSION);
    params0.setOnes();

    for (int i = 0; i < n_data; i++) 
        y_data(i) = params0[0] + params0[1]*u[i] + params0[2]*u[i]*u[i];
    
}

int main(int argc, char *argv[]) {
    std::cout << "***********************************************************" << std::endl;
    //-> create and fill synthetic data
    int n_data = 50;
    dvec u;
    dvec y_data, y_model;
    u.resize(n_data);
    y_data.resize(n_data);
    y_model.resize(n_data);

    create_synthethic_data(u, y_data, n_data);
    // <-

    
    // -> evaluation function
    auto evaluate = [&](dvec &params, int n_params) {
        for (int i = 0; i < n_data; i++) 
            y_model[i] = params[0] + params[1]*u[i] + params[2]*u[i]*u[i];
    };
    // <-

    // -> cost function
    CMAES::Engine::cost_type cost_func = [&](dvec &params, dvec &params_typical, int n_params) {
        
        evaluate(params, n_params);

        return (y_data - y_model).squaredNorm();
    };
    // <-

    // -> transform-scale-shift function
    CMAES::Engine::tss_type transform_scale_shift = [](Eigen::Ref<dvec> x, dvec &x_typical, dvec &x_tss, int n_params) {
        for (int i = 0; i < n_params; i++) {
            x_tss[i] = std::abs(x[i]);
        }
    };
    // <-

    CMAES::Engine cmaes;
    dvec x0(N_DIMENSION);
    x0.setZero();
    double sigma0 = 1;
    Solution sol = cmaes.fmin(x0, N_DIMENSION, sigma0, 0, 9999, cost_func, transform_scale_shift);

    return 0;
}