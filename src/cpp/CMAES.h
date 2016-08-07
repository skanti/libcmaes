#ifndef LIBCMAES_CMAES_H
#define LIBCMAES_CMAES_H

#include "Data.h"
#include "Types.h"
#include "Model.h"
#include "Parameters.h"
#include "mkl_vsl.h"
//#include "GnuplotIostream.h"


class CMAES {
public:
    typedef void (*tss_type)(double *, double *, double *, int);

    CMAES(Data *data_, Model *model_);

    dvec fmin(dvec &x0_, double sigma0_, dvec &x_typical_, int n_restarts, int seed, tss_type tss_func);

    void sample_offsprings();

    void rank_and_sort();

    void assign_new_mean();

    void cummulative_stepsize_adaption();

    void update_weights();

    void update_sigma();

    void update_cov_matrix();

    void eigendecomposition();

    void stopping_criteria();

    void update_best();

    double cost_function(double *params);

    void plot(dvec &params);

    void optimize();

    tss_type transform_scale_shift;

    Data *data;
    Model *model;
    Parameters era;

    // initial values
    dvec x0;
    dvec x_typical;
    double sigma0;
    int n_offsprings0;
    int n_params;

    // restarts
    int i_run;
    dvec params_best;
    double f_best;
    int n_offsprings;

    // cost
    int i_func_eval_tot;

    // random
    VSLStreamStatePtr rnd_stream;

    // options
    const int n_iteration_max = 5000;
    const int n_interval_plot = 100;
    bool should_stop_run;
    bool should_stop_optimization;

    // plotting
    //Gnuplot gp;

};


#endif //LIBCMAES_CMAES_H
