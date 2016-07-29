#include "CMAES.h"
#include <thread>
#include "SolverPool.h"
#include "mkl.h"

CMAES::CMAES(Data *data_, Model *model_)
        : data(data_), model(model_), dist_normal_real(0, 1), dist_uniform_real(0, 1) {
};

void CMAES::optimize() {
    std::cout << "\noptimization starting wtih: "
              << "n_offsprings: " << era.n_offsprings
              << " sigma: " << era.sigma
              << " f_best: " << f_best << std::endl;
    should_stop_run = false;
    while (era.i_iteration < n_iteration_max && !should_stop_run) {
        sample_offsprings();
        rank_and_sort();
        assign_new_mean();
        update_best();
        cummulative_stepsize_adaption();
        update_weights();
        update_cov_matrix();
        eigendecomposition();
        update_sigma();
        stopping_criteria();
        if (era.i_iteration % n_interval_plot == 0) {
            plot(era.params_mean);
        }

        era.i_iteration++;
    }
}

void CMAES::sample_offsprings() {
    for (int j = 0; j < era.n_params; j++) {
        for (int i = 0; i < era.n_offsprings; i++) {
            era.z_offsprings(i, j) = dist_normal_real(mt);
        }
    }
    SolverPool::dgemm(era.B.memptr(), era.D.memptr(), era.BD.memptr(), era.B.n_rows, era.B.n_cols, era.D.n_cols,
                      era.B.n_rows, era.C.n_rows, era.BD.n_rows);
    for (int i = 0; i < era.n_offsprings; i++) {
        SolverPool::dgemv(era.BD.memptr(), era.z_offsprings.memptr(), era.y_offsprings.memptr(), era.BD.n_rows,
                          era.BD.n_cols, era.BD.n_rows, 0, 0);
        std::copy(era.params_mean.data(), era.params_mean.data(), era.params_offsprings.get_col(i));
        SolverPool::daxpy(era.y_offsprings.memptr(), era.params_offsprings.get_col(i), era.sigma, era.n_params);
    }
}

void CMAES::rank_and_sort() {
    // -> rank by cost-function
    for (int i = 0; i < era.n_offsprings; i++) {
        era.f_offsprings[i] = cost_function(era.params_offsprings.get_col(i));
    }
    era.i_func_eval += era.n_offsprings;
    i_func_eval_tot += era.n_offsprings;
    // <-

    // -> sorting
    std::iota(era.keys_offsprings.begin(), era.keys_offsprings.end(), 0);
    std::sort(era.keys_offsprings.begin(), era.keys_offsprings.end(),
              [&](std::size_t idx1, std::size_t idx2) { return era.f_offsprings[idx1] < era.f_offsprings[idx2]; });
    for (int i = 0; i < era.n_parents; i++) {
        int idx_new = era.keys_offsprings[i];
        std::copy(era.params_offsprings.get_col(idx_new), era.params_offsprings.get_col(idx_new) + era.n_params,
                  era.params_parents_ranked.get_col(i));
    }
    for (int i = 0; i < era.n_offsprings; i++) {
        int idx_new = era.keys_offsprings[i];
        std::copy(era.y_offsprings.get_col(idx_new), era.y_offsprings.get_col(idx_new) + era.n_params,
                  era.y_offsprings_ranked.get_col(i));
    }
    // <-
}

void CMAES::update_best() {
    double f_cand = cost_function(era.params_parents_ranked.get_col(0));
    if (!std::isnan(f_cand) && f_cand < f_best) {
        std::copy(era.params_parents_ranked.get_col(0), era.params_parents_ranked.get_col(0) + era.n_params,
                  params_best.begin());
        f_best = f_cand;
    }
}

void CMAES::assign_new_mean() {
    std::copy(era.params_mean.begin(), era.params_mean.end(), era.params_mean_old.begin());
    std::fill(era.y_mean.begin(), era.y_mean.end(), 0.0);
    std::fill(era.params_mean.begin(), era.params_mean.end(), 0.0);
    SolverPool::mean_vector(era.params_parents_ranked.memptr(), era.n_params, era.n_parents, era.n_params, era.w.data(),
                            era.y_mean.data());
    SolverPool::mean_vector(era.params_parents_ranked.memptr(), era.n_params, era.n_parents, era.n_params, era.w.data(),
                            era.params_mean.data());

}

void CMAES::cummulative_stepsize_adaption() {
    // -> p sigma
    std::fill(era.p_s.begin(), era.p_s.end(), 0.0);
    SolverPool::dgemv(era.C_invsqrt.memptr(), era.y_mean.data(), era.p_s.data(), era.n_params, era.n_params,
                      era.n_params, era.p_s_fact, 1.0 - era.c_s);
    // <-

    // -> h sigma
    double p_s_norm = SolverPool::dnrm2(era.n_params, era.p_s.data());
    double p_s_thresh = (1.4 + 2.0 / (era.n_params + 1))
                        * std::sqrt(1.0 - std::pow(1.0 - era.c_s, 2.0 * (era.i_iteration + 1))) * era.chi;
    era.h_sig = p_s_norm < p_s_thresh;
    // <-

    // -> p cov
    std::fill(era.p_c.begin(), era.p_c.end(), 0.0);
    SolverPool::daxpy(era.p_c.data(), era.p_c.data(), 1.0 - era.c_c, era.n_params);
    SolverPool::daxpy(era.y_mean.data(), era.p_c.data(), era.h_sig * era.p_c_fact, era.n_params);
    // <-
}

void CMAES::update_weights() {

    for (int i = 0; i < era.n_offsprings; i++) {
        if (era.w[i] < 0) {
            SolverPool::dgemv(era.C_invsqrt.memptr(), era.y_offsprings_ranked.get_col(i), era.c_invsqrt_y.data(),
                              era.n_params, era.n_params, era.n_params, 1.0, 0.0);
            double h = SolverPool::dnrm2(era.n_params, era.c_invsqrt_y.data());
            era.w_var[i] = era.w[i] * era.n_params / (h * h);
        }
    }
}

void CMAES::update_sigma() {
    era.sigma *= std::exp(era.c_s / era.d_s * (SolverPool::dnrm2(era.n_params, era.p_s.data()) / era.chi - 1.0));
}

void CMAES::update_cov_matrix() {
    /*
    double h1 = (1 - era.h_sig) * era.c_c * (2.0 - era.c_c);
    dmat h2(era.n_params, era.n_params, arma::fill::zeros);
    for (int i = 0; i < era.n_offsprings; i++) {
        h2 += era.w_var[i] * era.y_offsprings_ranked[i] * era.y_offsprings_ranked[i].t();
    }
    era.C = (1.0 + era.c_1 * h1 - era.c_1 - era.c_mu * arma::sum(era.w)) * era.C
            + era.c_1 * era.p_c * era.p_c.t() + era.c_mu * h2;
            */
}

void CMAES::eigendecomposition() {
    //std::cout << era.C << std::endl;
    //era.C = arma::symmatu(era.C);
    //arma::eig_sym(era.C_eigvals, era.B, era.C);
    //dvec D2 = arma::sqrt(era.C_eigvals);
    //era.D = arma::diagmat(D2);
    //era.C_invsqrt = era.B * arma::inv(era.D) * era.B.t();
}

void CMAES::stopping_criteria() {
    double eigval_min = era.C_eigvals[0];
    double eigval_max = era.C_eigvals[era.n_params - 1];
    if (eigval_max / eigval_min > 1e14) {
        std::cout << "stopping criteria occured: bad covariance condition." << std::endl;
        std::cout << "stopping at iteration: " << era.i_iteration << std::endl;
        should_stop_run = true;
    }

    if (f_best < 1e-10) {
        std::cout << "stopping criteria occured: f_best small." << std::endl;
        std::cout << "stopping at iteration: " << era.i_iteration << std::endl;
        should_stop_run = true;
        should_stop_optimization = true;
    }
}

void CMAES::transform_scale_shift(double *params, double *params_tss) {
    SolverPool::transform_scale_shift(params, x_typical.data(), 0, 100, 1e-4, 1e-1, n_params, params_tss);
}

double CMAES::cost_function(double *params) {
    dvec params_tmp(n_params);
    transform_scale_shift(params, params_tmp.data());
    model->evaluate(data->x, params_tmp);
    double cost = 0.0;
    for (int i = 0; i < model->dim; i++) {
        cost += SolverPool::least_squares(model->y.get_col(i), data->y.get_col(i), data->n_data);
    }
    return cost;
}

void CMAES::plot(dvec &params) {
    //cost_function(params);
    //gp << "set yrange [] reverse\n";
    //gp << "plot '-' with points pt 7 title 'data', " << "'-' with lines title 'model'\n";
    //gp.send1d(boost::make_tuple(data->y[0], data->y[1]));
    //gp.send1d(boost::make_tuple(model->y[0], model->y[1]));
    //std::this_thread::sleep_for((std::chrono::nanoseconds) ((int) (0.025e9)));
}

void CMAES::fmin(dvec &x0_, double sigma0_, dvec &x_typical_, int n_restarts, int seed) {
    // -> settings
    mt.seed(seed);
    n_params = model->n_params;
    i_run = 0;
    // <-

    // -> first guess
    x0.resize(n_params);
    x0 = x0_;
    sigma0 = sigma0_;
    x_typical.resize(n_params);
    x_typical = x_typical_;
    // <-

    // -> prepare fmin
    params_best.resize(n_params);
    params_best = x0;
    f_best = std::isnan(cost_function(params_best.data())) ? std::numeric_limits<double>::infinity()
                                                           : cost_function(params_best.data());
    i_func_eval_tot = 0;
    int budget[2] = {0, 0};
    // <-

    // -> first run
    n_offsprings0 = (int) (4 + 3 * std::log(n_params));
    n_offsprings = n_offsprings0;
    era.init(n_offsprings, n_params, x0, sigma0);
    optimize();
    budget[0] += era.i_func_eval;
    // <-

    // -> restarts
    should_stop_optimization = false;
    for (i_run = 1; i_run < n_restarts + 1 && !should_stop_optimization; i_run++) {
        int n_regime1 = n_offsprings0 * (1 << i_run);
        while (budget[0] > budget[1]) {
            double ur2 = std::pow(dist_uniform_real(mt), 2.0);
            int n_offsprings = (int) (n_offsprings0 * std::pow(0.5 * n_regime1 / n_offsprings0, ur2));
            double us = dist_uniform_real(mt);
            double sigma_regime2 = sigma0 * 2.0 * std::pow(10, -2.0 * us);
            era.init(n_offsprings, n_params, params_best, sigma_regime2);
            optimize();
            budget[1] += era.i_func_eval;
        }
        n_offsprings = n_regime1;
        era.init(n_offsprings, n_params, params_best, sigma0);
        optimize();
        budget[0] += era.i_func_eval;
        std::cout << "i_run: " << i_run << " / " << n_restarts << " completed." << std::endl;
    }
    // <-

    // -> plot final result
    cost_function(params_best.data());
    plot(params_best);
    dvec params_best_unscaled(n_params);
    transform_scale_shift(params_best.data(), params_best_unscaled.data());
    std::cout << "f_best: " << f_best << ", params_best: ";
    for (int i = 0; i < era.n_params; i++)
        std::cout << params_best_unscaled[i] << " ";
    // <-
}

