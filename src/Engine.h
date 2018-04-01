#pragma once

#include <functional>
#include "Common.h"
#include "Parameter.h"

namespace CMAES {
	typedef std::function<double(const double *, int)> cost_type;
	typedef std::function<void(double *, int)> transform_type;
    
	class Engine {
    public:

        Solution fmin(dvec &x0_, int n_params_, double sigma0_, int n_restarts, int seed, cost_type costf, transform_type ftransform);

    private:
        transform_type ftransform;

        cost_type fcost;

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

        double cost(Eigen::Ref<dvec> params);

        Parameter era; // <- instance of Parameter class.

        Eigen::EigenSolver<dmat> eigensolver;

        // -> random 
        std::mt19937 mt;
        std::normal_distribution<double> dist_normal_real;
        std::uniform_real_distribution<double> dist_uniform_real;        
        // <-


        dvec x0; // <- user provided guess.
        dvec x_typical; // <- typical order.
        double sigma0; // <- step-size.
        int n_params; // <- dimensionality of x.
        int n_offsprings;
        int n_parents;

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
