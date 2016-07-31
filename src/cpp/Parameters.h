#ifndef LIBCMAES_PARAMETERS_H
#define LIBCMAES_PARAMETERS_H

#include <vector>
#include <cmath>
#include <algorithm>
#include "Types.h"

struct Parameters {

    void init(int n_offsprings_, int n_params_, dvec &params_mean_, double &sigma_);

    int n_offsprings;
    int n_params;
    int n_parents;
    int i_iteration;
    int i_func_eval;
    double n_mu_eff;
    dmat params_offsprings;
    dmat params_parents_ranked;
    dmat z_offsprings;
    dmat y_offsprings;
    dmat y_offsprings_ranked;
    dvec f_offsprings;
    dvec w;
    dvec w_var;
    dvec c_invsqrt_y;
    dvec y_mean;
    dvec params_mean;
    dvec params_mean_old;
    ivec keys_offsprings;
    double a_mu;
    double a_mueff;
    double a_posdef;
    double c_c;
    double c_s;
    double c_1;
    double c_mu;
    double c_m;
    double d_s;
    double chi;
    dvec p_c;
    dvec p_s;
    double p_c_fact;
    double p_s_fact;
    double sigma;
    dvec C_eigvals;
    dvec C_eigvals2;
    dmat C;
    dmat C_invsqrt;
    dmat C_invsqrt_tmp;
    dmat B;
    dmat D;
    dmat D_inv;
    dmat BD;
    bool h_sig;
};


#endif //LIBCMAES_PARAMETERS_H
