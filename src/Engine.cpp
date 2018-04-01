#include "Engine.h"
#include <cmath>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include "Timer.h"

namespace CMAES {

    void Engine::optimize() {
#ifndef NDEBUG
        std::cout << "\nstarting run. "
                  << "n_offsprings: " << era.n_offsprings
                  << " sigma: " << era.sigma
                  << " f_best: " << f_best << std::endl;
#endif
        should_stop_run = false;
        while (era.i_iteration < n_iteration_max && !should_stop_run) {
            sample_offsprings();
            rank_and_sort();
            update_best();
            assign_new_mean();
            update_evolution_paths();
            update_weights();
            update_cov_matrix();
            eigendecomposition();
            update_stepsize();
            stopping_criteria();

            era.i_iteration++;
        }
    }

    void Engine::sample_offsprings() {
        
        for (int j = 0; j < n_offsprings; j++) {
            for (int i = 0; i < n_params; i++) {
                era.z_offsprings(i, j) = dist_normal_real(mt);
            }
        }

        dmat BD(n_params, n_params);
        BD.block(0, 0, n_params, n_params) = era.B*era.D.block(0, 0, n_params, n_params);
        
        era.y_offsprings.block(0, 0, n_params, n_offsprings) = BD*era.z_offsprings.block(0, 0, n_params, n_offsprings);

        for (int i = 0; i < era.n_offsprings; i++) {
            era.x_offsprings.col(i) = era.x_mean + era.sigma*era.y_offsprings.col(i);
        }
    }

    void Engine::rank_and_sort() {
        // -> rank by cost-function
        for (int i = 0; i < era.n_offsprings; i++) {
            double f_cand = fcost(era.x_offsprings.col(i).data(), era.n_params);
            era.f_offsprings[i] = std::isnan(f_cand) ? std::numeric_limits<double>::infinity() : f_cand;
        }
        
        era.i_func_eval += n_offsprings;
        i_func_eval_tot += n_offsprings;
        // <-

        
        // -> sorting
        std::iota(era.keys_offsprings.data(), era.keys_offsprings.data() + era.n_offsprings, 0);
        std::sort(era.keys_offsprings.data(), era.keys_offsprings.data() + era.n_offsprings, [&](int i, int j) { return era.f_offsprings[i] < era.f_offsprings[j];});
        
        for (int i = 0; i < era.n_parents; i++) {
            int i1 = era.keys_offsprings[i];
            era.x_parents_ranked.col(i) = era.x_offsprings.col(i1);
        }

        for (int i = 0; i < era.n_offsprings; i++) {
            int i1 = era.keys_offsprings[i];
            era.y_offsprings_ranked.col(i) = era.y_offsprings.col(i1);
        }
        // <-
    }

    void Engine::update_best() {
        double f_cand = fcost(era.x_parents_ranked.col(0).data(), era.n_params);
        if (!std::isnan(f_cand) && f_cand < f_best) {
            x_best = era.x_parents_ranked.col(0);
            f_best = f_cand;
        }
    }

    void Engine::assign_new_mean() {
        
        era.x_mean_old = era.x_mean;

        era.y_mean = era.y_offsprings_ranked.block(0, 0, n_params, n_parents)*era.w.block(0, 0, n_parents, 1);
        era.x_mean = era.x_mean + era.sigma*era.y_mean;
    }

    void Engine::update_evolution_paths() {
        
        // -> p sigma
        era.p_s = (1.0 - era.c_s)*era.p_s + era.p_s_fact*era.C_invsqrt*era.y_mean;
        // <-

        // -> h sigma
        double p_s_norm =era.p_s.norm();
        double p_s_thresh = (1.4 + 2.0 / (era.n_params + 1))*std::sqrt(1.0 - std::pow(1.0 - era.c_s, 2.0*(era.i_iteration + 1)))*era.chi;
        era.h_sig = p_s_norm < p_s_thresh;
        // <-

        // -> p cov
        era.p_c = (1.0 - era.c_c)*era.p_c + era.h_sig*era.p_c_fact*era.y_mean;
        // <-
    }

    void Engine::update_weights() {
        // dmat c_invsqrt_y(n_params, n_params);
        for (int i = 0; i < era.n_offsprings; i++) {
            if (era.w[i] < 0) {
                era.w_var[i] = era.w[i]*era.n_params/(era.C_invsqrt*era.y_offsprings_ranked.col(i)).squaredNorm();
            }
        }
    }

    void Engine::update_cov_matrix() {
        double h1 = (1 - era.h_sig)*era.c_c*(2.0 - era.c_c);

        Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> W(era.w_var.block(0, 0, n_offsprings, 1));
        double w_sum = era.w.block(0, 0, n_offsprings, 1).sum();
        era.C = (1.0 + era.c_1*h1 - era.c_1 - era.c_mu*w_sum)*era.C + era.c_1*era.p_c*era.p_c.transpose()
              + era.c_mu*(era.y_offsprings_ranked.block(0, 0, n_params, n_offsprings)*W*era.y_offsprings_ranked.block(0, 0, n_params, n_offsprings).transpose());

    }

    void Engine::update_stepsize() {
        era.sigma *= std::exp(era.c_s / era.d_s*(era.p_s.norm()/era.chi - 1.0));
    }

    void Engine::eigendecomposition() {
        eigensolver.compute(era.C);
        era.eigvals_C = eigensolver.eigenvalues().real(); 
        era.D = era.eigvals_C.cwiseSqrt().asDiagonal();
        era.B = dmat(eigensolver.eigenvectors().real());
        dmat D_inv = era.eigvals_C.cwiseSqrt().cwiseInverse().asDiagonal();
        era.C_invsqrt = era.B*D_inv*era.B.transpose();
    }

    void Engine::stopping_criteria() {
        double eigval_min = era.eigvals_C[0];
        double eigval_max = era.eigvals_C[era.n_params - 1];

        // -> condition of covariance matrix
        if (eigval_max / eigval_min > 1e14) {
#ifndef NDEBUG
            std::cout << "stopping criteria. iteration: " << era.i_iteration << " reason: bad covariance condition."
                      << std::endl;
#endif
            should_stop_run = true;
        }
        // <-

        // -> cost function already low enough
        if (f_best < 1e-10) {
#ifndef NDEBUG
            std::cout << "stopping criteria. iteration: " << era.i_iteration << " reason: f_best small." << std::endl;
#endif
            should_stop_run = true;
            should_stop_optimization = true;
        }
        // <-

        // -> sigma up tolerance
        double sigma_fac = era.sigma / sigma0;
        double sigma_up_thresh = 1e20*std::sqrt(eigval_max);
        if (sigma_fac / sigma0 > sigma_up_thresh) {
#ifndef NDEBUG
            std::cout << "stopping criteria. iteration: " << era.i_iteration << " reason: sigma up." << std::endl;
#endif
            should_stop_run = true;
        }
        // <-

        // -> no effect axis
        int nea = 0;
        for (int i = 0; i < era.n_params; i++) {
            double ei = 0.1*era.sigma*era.eigvals_C[i];
            for (int j = 0; j < era.n_params; j++) {
                nea += era.x_mean[i] == era.x_mean[i] + ei*era.B(j, i);
            }
        }
        if (nea > 0) {
#ifndef NDEBUG
            std::cout << "stopping criteria. iteration: " << era.i_iteration << " reason: no effect axis." << std::endl;
#endif
            should_stop_run = true;
        }
        // <-

        // -> no effect coordinate
        int nec = 0;
        for (int i = 0; i < era.n_params; i++) {
            nec += era.x_mean[i] == era.x_mean[i] + 0.2*era.sigma*std::sqrt(era.C(i, i));
        }
        if (nec > 0) {
#ifndef NDEBUG
            std::cout << "stopping criteria. iteration: " << era.i_iteration << " reason: no effect coordinate."
                      << std::endl;
#endif
            should_stop_run = true;
        }
        // <-
    }

    void Engine::print(dvec &params) {
#ifndef NDEBUG
        std::cout << "f0: " << std::setprecision(std::numeric_limits<double>::digits10 + 1) << era.f_offsprings[0] << std::endl;
#endif
    }

	Solution Engine::fmin(dvec &x0_, int n_params_, double sigma0_, int n_restarts, int seed, cost_type fcost0, transform_type ftransform0) {
        fcost = fcost0;
        ftransform = ftransform0;
        mt.seed(seed);

        // -> settings
        n_params = n_params_;
        i_run = 0;
        // <-

        
        // -> first guess
        x0 = x0_;
        sigma0 = sigma0_;
        // <-

        // -> prepare fmin
        int n_offsprings0 = (int) (4 + 3*std::log(n_params));
        n_offsprings = n_offsprings0;
        n_parents = n_offsprings/2;
        int n_offsprings_max = n_offsprings0*(1 << n_restarts);
        era.reserve(n_offsprings_max, n_offsprings_max/2, n_params);
        era.reinit(n_offsprings, n_parents, n_params, x0, sigma0);
        x_best = x0;
        double f_cand = fcost(x_best.data(), n_params);
        f_best = std::isnan(f_cand) ? std::numeric_limits<double>::infinity() : f_cand;
        i_func_eval_tot = 0;
        int budget[2] = {0, 0};
        // <-

        // -> first run
        optimize();
        budget[0] += era.i_func_eval;
        // <-

        // -> restarts
        for (i_run = 1, should_stop_optimization = 0; i_run < n_restarts + 1 && !should_stop_optimization; i_run++) {
            int n_regime1 = n_offsprings0*(1 << i_run);
            while (budget[0] > budget[1] && !should_stop_optimization) {
                double ur2 = std::pow(dist_uniform_real(mt), 2.0);
                n_offsprings = (int) (n_offsprings0*std::pow(0.5*n_regime1 / n_offsprings0, ur2));
                n_parents = n_offsprings/2;
                double sigma_regime2 = sigma0*2.0*std::pow(10, -2.0*dist_uniform_real(mt));
                era.reinit(n_offsprings, n_parents, n_params, x_best, sigma_regime2);
                optimize();
                budget[1] += era.i_func_eval;
            }
            n_offsprings = n_regime1;
            n_parents = n_offsprings/2;
            era.reinit(n_offsprings, n_parents, n_params, x0, sigma0);
            optimize();
            budget[0] += era.i_func_eval;
            std::cout << "i_run: " << i_run << " / " << n_restarts << " completed. f_best: " << f_best << std::endl;
        }
        // <-
        

        // -> print final result
        fcost(x_best.data(), n_params);
        print(x_best);
        ftransform(x_best.data(), n_params);
        // <-
        return Solution{x_best, f_best};

    }

}
