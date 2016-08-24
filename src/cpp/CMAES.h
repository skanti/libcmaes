#ifndef LIBCMAES_CMAES_H
#define LIBCMAES_CMAES_H

#include "World.h"
#include "Types.h"
#include "Parameters.h"
#include "mkl_vsl.h"
//#include "GnuplotIostream.h"


class CMAES {
public:
    typedef void (*tss_type)(double *, double *, double *, int);

    CMAES(World *data_);

    dvec fmin(dvec &params_typical_, double sigma0_, int n_restarts, int seed, tss_type tss_);

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

    void plot(dvec &params);

    void optimize();

    double cost(double *params);

    tss_type transform_scale_shift;

    World *world;
    Parameters era;

    // initial values
    dvec params0;
    dvec params_typical;
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
