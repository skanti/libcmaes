#pragma once

#include <functional>
#include "Common.h"
#include "Parameter.h"

namespace CMAES {
    class Engine {
    public:
        typedef void (*tss_type)(double *, double *, double *, int);

        typedef std::function<double(dvec &, dvec &, int)> cost_type;

        Solution
        fmin(dvec &x0_, int n_params_, double sigma0_, int n_restarts, int seed, cost_type costf, tss_type tssf);

    private:
        tss_type transform_scale_shift;

        cost_type cost_func;

        void sample_offsprings();

        void rank_and_sort();

        void assign_new_mean();

        void update_evolution_paths();

        void update_weights();

        void update_stepsize();

        void update_cov_matrix();

        void eigendecomposition();

        void stopping_criteria();

        void update_best();

        void print(dvec &params);

        void optimize();

        double cost(double *params);

        Parameter era; // <- instance of Parameter class.

        dvec x0; // <- user provided guess.
        dvec x_typical; // <- typical order.
        double sigma0; // <- step-size.
        int n_params; // <- dimensionality of x.

        int i_run; // <- restart counter.
        dvec x_best; // <- current best estimate.
        double f_best; // <- cost of x_best.
        int i_func_eval_tot; // <- function evaluation counter.

        const int n_iteration_max = 5000; // <- max number of function evals.
        const int n_interval_print = 100; // <- print (cout) interval.
        bool should_stop_run; // <- generation stoping flag.
        bool should_stop_optimization; // <- optimization stoping flag.

    };
}