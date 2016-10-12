#ifndef CMAES_ENGINE_H
#define CMAES_ENGINE_H

#include "World.h"
#include "CMAESTypes.h"
#include "Parameter.h"
#include "mkl_vsl.h"

namespace CMAES {
    class Engine {
    public:
        Engine(World *data_);

        Solution fmin(dvec &x_typical_, double sigma0_, int n_restarts, int seed);

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

        void plot(dvec &params);

        void optimize();

        double cost(double *params);

        World *world;
        Parameter era;

        // initial values
        dvec x0;
        dvec x_typical;
        double sigma0;
        int n_offsprings0;
        int n_params;

        // restarts
        int i_run;
        dvec x_best;
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
}

#endif
