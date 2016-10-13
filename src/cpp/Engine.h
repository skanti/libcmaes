#ifndef CMAES_ENGINE_H
#define CMAES_ENGINE_H

#include "CMAESTypes.h"
#include "Parameter.h"
#include "mkl_vsl.h"
#include <functional>

namespace CMAES {
    class Engine {
    public:
        typedef std::function<void(double *, double *, int)> tss_type;

        typedef std::function<double(dvec &, int)> cost_type;

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

        double cost(double *x);

        Parameter era; // <- instance of Parameter class.

        dvec x0; // <- user provided guess.
        dvec x_typical; // <- typical order.
        double sigma0; // <- step-size.
        int n_params; // <- dimensionality of x.

        int i_run; // <- restart counter.
        dvec x_best; // <- current best estimate.
        double f_best; // <- cost of x_best.
        int i_func_eval_tot; // <- function evaluation counter.

        VSLStreamStatePtr rnd_stream; // <- random number stream

        const int n_iteration_max = 5000; // <- max number of function evals.
        const int n_interval_print = 100; // <- print (cout) interval.
        bool should_stop_run; // <- generation stoping flag.
        bool should_stop_optimization; // <- optimization stoping flag.

    };
}

#endif
