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

    void prepare_optimization();

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

    void plot();

    void optimize();

    Data *data;
    Model *model;
    Parameters era;

    // initial user guesses
    dvec x0;
    double sigma0;

    // restarts
    int i_run;
    dvec params_best;
    double f_best;
    double fac_inc_pop;
    int n_offsprings;

    //
    std::mt19937 mt;
    std::normal_distribution<double> dist_normal;

    // options
    const int n_iteration_max = (int) (1e6);
    const int n_interval_plot = 200;
    bool should_stop;

    // plotting
    Gnuplot gp;

};


#endif //LIBCMAES_CMAES_H
