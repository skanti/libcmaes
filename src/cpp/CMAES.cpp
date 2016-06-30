#include "CMAES.h"
#include <thread>

CMAES::CMAES(Data *data_, Model *model_)
        : data(data_), model(model_), dist_normal() {
};

void CMAES::prepare_optimization() {
    int n_params = model->n_params;
    if (i_run == 0)
        n_offsprings = (int) (4 + 3 * std::log(n_params));
    else
        n_offsprings *= fac_inc_pop;

    era.init(n_offsprings, n_params);
}

void CMAES::optimize() {
    should_stop = false;
    while (era.i_iteration < n_iteration_max && !should_stop) {
        sample_offsprings();
        rank_and_sort();
        assign_new_mean();
        update_best();
        cummulative_stepsize_adaption();
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
        std::generate(zo.begin(), zo.end(), [&]() { return dist_normal(mt); });

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
    // <-

    // -> sorting
    std::iota(era.keys_offsprings.begin(), era.keys_offsprings.end(), 0);
    std::sort(era.keys_offsprings.begin(), era.keys_offsprings.end(),
              [&](std::size_t idx1, std::size_t idx2) { return era.f_offsprings[idx1] < era.f_offsprings[idx2]; });
    for (int i = 0; i < era.n_parents; i++) {
        int idx_new = era.keys_offsprings[i];
        era.y_parents[i] = era.y_offsprings[idx_new];
        era.params_parents[i] = era.params_offsprings[idx_new];
    }
    // <-
}

void CMAES::update_best() {
    double f_mean = cost_function(era.params_mean);
    if (!std::isnan(f_mean) && f_mean < f_best) {
        params_best = era.params_mean;
        f_best = f_mean;
    }
}

void CMAES::assign_new_mean() {
    era.params_mean_old = era.params_mean;
    era.params_mean.zeros();
    era.y_mean.zeros();
    for (int i = 0; i < era.n_parents; i++) {
        era.y_mean += era.y_parents[i] * era.w_parents[i];
        era.params_mean += era.params_parents[i] * era.w_parents[i];
    }
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

void CMAES::update_sigma() {
    era.sigma *= std::exp(era.c_s / era.d_s * (arma::norm(era.p_s) / era.chi - 1.0));
}

void CMAES::update_cov_matrix() {
    double h1 = (1 - era.h_sig) * era.c_c * (2.0 - era.c_c);
    dmat h2(era.n_params, era.n_params);
    h2.zeros();
    for (int i = 0; i < era.n_parents; i++) {
        h2 += era.w_parents[i] * era.y_parents[i] * era.y_parents[i].t();
    }
    era.C = (1.0 + era.c_1 * h1 - era.c_1 - era.c_mu * arma::sum(era.w_parents)) * era.C
            + era.c_1 * era.p_c * era.p_c.t() + era.c_mu * h2;
}

void CMAES::eigendecomposition() {
    era.C = arma::symmatu(era.C);
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
    return arma::norm(model->y[0] - data->y[0]);
}

void CMAES::plot() {
    cost_function(era.params_mean);
    gp << "plot '-' with lines title 'model', " << "'-' with points pt 7 title 'data'\n";
    gp.send1d(boost::make_tuple(data->x, model->y[0]));
    gp.send1d(boost::make_tuple(data->x, data->y[0]));
    std::this_thread::sleep_for((std::chrono::nanoseconds) ((int) (0.4e9)));
}

void CMAES::fmin(dvec &x0_, double sigma0_, int n_restarts, int seed) {
    mt.seed(seed);
    // -> first guess
    x0.resize(model->n_params);
    x0 = x0_;
    sigma0 = sigma0_;
    // <-

    // -> prepare fmin
    params_best.resize(model->n_params);
    params_best = x0;
    f_best = std::isnan(cost_function(params_best)) ? std::numeric_limits<double>::infinity() : cost_function(
            params_best);
    fac_inc_pop = 2.0;
    // <-

    // -> runs
    era.params_mean = x0;
    era.sigma = sigma0;
    for (i_run = 0; i_run < n_restarts + 1; i_run++) {
        prepare_optimization();
        optimize();
        era.params_mean = params_best;
        era.sigma = sigma0;
        std::cout << "i_run: " << (i_run + 1) << " / " << (n_restarts + 1) << " completed." << std::endl;
    }
    // <-

    // -> plot final result
    cost_function(params_best);
    plot();
    std::cout << params_best << std::endl;
    // <-
}

