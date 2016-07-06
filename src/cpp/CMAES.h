#ifndef LIBCMAES_CMAES_H
#define LIBCMAES_CMAES_H

#include "Data.h"
#include "Types.h"
#include "Model.h"
#include "Parameters.h"
#include "GnuplotIostream.h"


class CMAES {
public:
    CMAES(Data *data_, Model *model_);

    void fmin(dvec &x0_, double sigma0_, int n_restarts, int seed = 999);

    void sample_offsprings();

    void rank_and_sort();

    void assign_new_mean();

    void cummulative_stepsize_adaption();

    void update_weights();

    void update_sigma();

    void update_cov_matrix();

    void eigendecomposition();

    void check_cov_matrix_condition();

    void update_best();

    double cost_function(dvec &params);

    void plot(dvec &params);

    void optimize();

    Data *data;
    Model *model;
    Parameters era;

    // initial values
    dvec x0;
    double sigma0;
    int n_offsprings0;
    int n_params;

    // bipop
    int n_offsprings_l;

    // restarts
    int i_run;
    dvec params_best;
    double f_best;
    double fac_inc_pop;
    int n_offsprings;

    // cost
    int i_func_eval_tot;

    // random
    std::mt19937 mt;
    std::normal_distribution<double> dist_normal_real;
    std::uniform_real_distribution<double> dist_uniform_real;

    // options
    const int n_iteration_max = (int) (1e6);
    const int n_interval_plot = 200;
    bool should_stop;

    // plotting
    Gnuplot gp;

};


#endif //LIBCMAES_CMAES_H
