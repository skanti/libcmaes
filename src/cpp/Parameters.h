#ifndef LIBCMAES_PARAMETERS_H
#define LIBCMAES_PARAMETERS_H

#include <vector>
#include <cmath>
#include <algorithm>
#include "Types.h"

struct Parameters {

    void init(int n_offsprings_, int n_params_);

    int n_offsprings;
    int n_params;
    int n_parents;
    int i_evals;
    int i_iteration;
    double n_parents_eff;
    std::vector<dvec> params_offsprings;
    std::vector<dvec> params_parents;
    std::vector<dvec> z_offsprings;
    std::vector<dvec> y_offsprings;
    std::vector<dvec> y_parents;
    dvec f_offsprings;
    dvec w_parents;
    dvec y_mean;
    dvec params_mean;
    dvec params_mean_old;
    arma::Col<int> keys_offsprings;
    double c_c;
    double c_s;
    double c_1;
    double c_mu;
    double d_s;
    double chi;
    dvec p_c;
    dvec p_s;
    double p_c_fact;
    double p_s_fact;
    double sigma;
    dvec C_eigvals;
    dmat C;
    dmat C_sym;
    dmat C_invsqrt;
    dmat B;
    dmat D;
    bool h_sig;
};


#endif //LIBCMAES_PARAMETERS_H
