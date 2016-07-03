#include "CMAES.h"
#include <thread>

CMAES::CMAES(Data *data_, Model *model_)
        : data(data_), model(model_), dist_normal_real() {
};

void CMAES::prepare_optimization() {
    dvec params_mean;
    double sigma;
    if (i_run == 0) {
        n_offsprings0 = (int) (4 + 3 * std::log(n_params));
        n_offsprings = n_offsprings0;
        n_offsprings_l = n_offsprings0;
        params_mean = x0;
        sigma = sigma0;
    } else {
        int n_regime1 = n_offsprings0 * (1 << i_run);
        double ur2 = std::pow(dist_uniform_real(mt), 2.0);
        int n_regime2 = (int) (n_offsprings0 * std::pow(0.5 * n_regime1 / n_offsprings0, ur2));
        double us = dist_uniform_real(mt);
        double sigma_regime2 = sigma0 * 2.0 * std::pow(10, -2.0 * us);
        if (n_regime1 <= n_regime2) {
            n_offsprings = n_regime1;
            n_offsprings_l = n_regime1;
            sigma = sigma0;
            std::cout << "regime 1" << std::endl;
        } else {
            n_offsprings = n_regime2;
            sigma = sigma_regime2;
            std::cout << "regime 2" << std::endl;
        }
        params_mean = params_best;
    }
    era.init(n_offsprings, n_params, params_mean, sigma);
}

void CMAES::optimize() {
    should_stop = false;
    while (era.i_iteration < n_iteration_max && !should_stop) {
        sample_offsprings();
        rank_and_sort();
        assign_new_mean();
        update_best();
        cummulative_stepsize_adaption();
        update_weights();
        update_cov_matrix();
        update_sigma();
        eigendecomposition();
        check_cov_matrix_condition();
        if (era.i_iteration % n_interval_plot == 0)
            plot();

        era.i_iteration++;
    }
}

void CMAES::sample_offsprings() {
    for (dvec &zo : era.z_offsprings)
        std::generate(zo.begin(), zo.end(), [&]() { return dist_normal_real(mt); });

    dmat BD = era.B * era.D;
    for (int i = 0; i < era.n_offsprings; i++) {
        era.y_offsprings[i] = BD * era.z_offsprings[i];
        era.params_offsprings[i] = era.params_mean + era.sigma * era.y_offsprings[i];
    }
}

void CMAES::rank_and_sort() {
    // -> ranking by cost-function
    for (int i = 0; i < era.n_offsprings; i++) {
        era.f_offsprings[i] = cost_function(era.params_offsprings[i]);
    }
    n_func_evals += era.n_offsprings;
    // <-
    // -> sorting
    std::iota(era.keys_offsprings.begin(), era.keys_offsprings.end(), 0);
    std::sort(era.keys_offsprings.begin(), era.keys_offsprings.end(),
              [&](std::size_t idx1, std::size_t idx2) { return era.f_offsprings[idx1] < era.f_offsprings[idx2]; });
    for (int i = 0; i < era.n_parents; i++) {
        int idx_new = era.keys_offsprings[i];
        era.y_parents_ranked[i] = era.y_offsprings[idx_new];
        era.params_parents_ranked[i] = era.params_offsprings[idx_new];
    }
    for (int i = 0; i < era.n_offsprings; i++) {
        int idx_new = era.keys_offsprings[i];
        era.y_offsprings_ranked[i] = era.y_offsprings[idx_new];
    }
    // <-
}

void CMAES::update_best() {
    double f_cand = cost_function(era.params_parents_ranked[0]);
    if (!std::isnan(f_cand) && f_cand < f_best) {
        params_best = era.params_parents_ranked[0];
        f_best = f_cand;
    }
}

void CMAES::assign_new_mean() {
    era.params_mean_old = era.params_mean;
    era.params_mean.zeros();
    era.y_mean.zeros();
    for (int i = 0; i < era.n_parents; i++) {
        era.y_mean += era.y_parents_ranked[i] * era.w[i];
        era.params_mean += era.params_parents_ranked[i] * era.w[i];
    }
    //std::cout << era.params_mean << std::endl;
}

void CMAES::cummulative_stepsize_adaption() {
    // -> p sigma
    era.p_s = (1.0 - era.c_s) * era.p_s + era.p_s_fact * era.C_invsqrt * era.y_mean;
    // <-

    // -> h sigma
    double p_s_norm = arma::norm(era.p_s);
    era.h_sig = p_s_norm / std::sqrt(1.0 - std::pow(1.0 - era.c_s, 2.0 * (era.i_iteration + 1)))
                < (1.4 + 2.0 / (era.n_params + 1)) * era.chi;
    // <-

    // -> p cov
    era.p_c = (1.0 - era.c_c) * era.p_c + era.h_sig * era.p_c_fact * era.y_mean;
    // <-
}

void CMAES::update_weights() {
    for (int i = 0; i < era.n_offsprings; i++) {
        if (era.w[i] < 0) {
            double h = arma::norm(era.C_invsqrt * era.y_offsprings_ranked[i]);
            era.w_var[i] = era.w[i] * era.n_params / (h * h);
        }
    }
}

void CMAES::update_sigma() {
    era.sigma *= std::exp(era.c_s / era.d_s * (arma::norm(era.p_s) / era.chi - 1.0));
}

void CMAES::update_cov_matrix() {
    double h1 = (1 - era.h_sig) * era.c_c * (2.0 - era.c_c);
    dmat h2(era.n_params, era.n_params, arma::fill::zeros);
    for (int i = 0; i < era.n_offsprings; i++) {
        h2 += era.w_var[i] * era.y_offsprings_ranked[i] * era.y_offsprings_ranked[i].t();
    }
    era.C = (1.0 + era.c_1 * h1 - era.c_1 - era.c_mu * arma::sum(era.w)) * era.C
            + era.c_1 * era.p_c * era.p_c.t() + era.c_mu * h2;
}

void CMAES::eigendecomposition() {
    //std::cout << era.C << std::endl;
    //era.C = arma::symmatu(era.C);
    arma::eig_sym(era.C_eigvals, era.B, era.C);
    dmat D2 = arma::diagmat(era.C_eigvals);
    era.D = arma::sqrt(D2);
    era.C_invsqrt = era.B * arma::inv(era.D) * era.B.t();
}

void CMAES::check_cov_matrix_condition() {
    auto eigval_minmax = std::minmax_element(era.C_eigvals.begin(), era.C_eigvals.end());
    if ((*eigval_minmax.second) / (*eigval_minmax.first) > 1e14) {
        std::cout << "stopping criteria occured: bad covariance condition." << std::endl;
        std::cout << "stopping at iteration: " << era.i_iteration << std::endl;

        should_stop = true;
    }
}

double CMAES::cost_function(dvec &params) {
    model->evaluate(data->x, params);
    double cost = 0.0;
    for (int i = 0; i < model->dim; i++)
        cost += arma::norm(model->y[i] - data->y[i]);
    return cost;
}

void CMAES::plot() {
    cost_function(era.params_mean);
    gp << "set yrange [] reverse\n";
    gp << "plot '-' with points pt 7 title 'data', " << "'-' with lines title 'model'\n";
    gp.send1d(boost::make_tuple(data->y[0], data->y[1]));
    gp.send1d(boost::make_tuple(model->y[0], model->y[1]));
    std::this_thread::sleep_for((std::chrono::nanoseconds) ((int) (0.2e9)));
}

void CMAES::fmin(dvec &x0_, double sigma0_, int n_restarts, int seed) {
    mt.seed(seed);
    n_params = model->n_params;
    // -> first guess
    x0.resize(n_params);
    x0 = x0_;
    sigma0 = sigma0_;
    // <-

    // -> prepare fmin
    params_best.resize(n_params);
    params_best = x0;
    f_best = std::isnan(cost_function(params_best)) ? std::numeric_limits<double>::infinity() : cost_function(
            params_best);
    fac_inc_pop = 2.0;
    n_func_evals = 0;
    // <-

    // -> runs
    for (i_run = 0; i_run < n_restarts + 1; i_run++) {
        prepare_optimization();
        optimize();
        std::cout << "params_best: " << params_best.t() << std::endl;
        std::cout << "i_run: " << (i_run + 1) << " / " << (n_restarts + 1) << " completed." << std::endl;
    }
    // <-

    // -> plot final result
    cost_function(params_best);
    plot();
    std::cout << params_best << std::endl;
    // <-
}

