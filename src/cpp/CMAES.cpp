#include "CMAES.h"
#include "MathKernels.h"
#include <iostream>
#include <numeric>
#include <iomanip>
#include "Timer.h"

CMAES::CMAES(Data *data_, Model *model_)
        : data(data_), model(model_) {
};

void CMAES::optimize() {
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
        cummulative_stepsize_adaption();
        update_weights();
        update_cov_matrix();
        eigendecomposition();
        update_sigma();
        stopping_criteria();
#ifndef NDEBUG
        if (era.i_iteration % n_interval_plot == 0) {
            plot(era.params_mean);
        }
#endif
        era.i_iteration++;
    }
}

void CMAES::sample_offsprings() {
    double BD[n_params * n_params] __attribute__((aligned(32)));
    MathKernels::sample_random_vars_gaussian(&rnd_stream, era.n_params * era.n_offsprings, era.z_offsprings.memptr(),
                                             0.0, 1.0);
    MathKernels::dgemm(era.B.memptr(), 0, era.D.memptr(), 0, BD, era.B.n_rows, era.B.n_cols, era.D.n_cols,
                       1.0, 0, era.B.n_rows, era.D.n_rows, n_params);

    MathKernels::dgemm(BD, 0, era.z_offsprings.memptr(), 0, era.y_offsprings.memptr(), n_params,
                       n_params, era.z_offsprings.n_cols, 1.0, 0, n_params, era.z_offsprings.n_rows,
                       era.y_offsprings.n_rows);
    for (int i = 0; i < era.n_offsprings; i++) {
        std::copy(era.params_mean.data(), era.params_mean.data() + era.n_params, era.params_offsprings.memptr(i));
        MathKernels::daxpy(era.y_offsprings.memptr(i), era.params_offsprings.memptr(i), era.sigma, era.n_params);
    }
}

void CMAES::rank_and_sort() {
    // -> rank by cost-function
//#pragma omp parallel for
    for (int i = 0; i < era.n_offsprings; i++) {
        double f_cand = cost_function(era.params_offsprings.memptr(i));
        era.f_offsprings[i] = std::isnan(f_cand) ? std::numeric_limits<double>::infinity() : f_cand;
    }
    era.i_func_eval += era.n_offsprings;
    i_func_eval_tot += era.n_offsprings;
    // <-

    // -> sorting
    std::iota(era.keys_offsprings.data(), era.keys_offsprings.data() + era.n_offsprings, 0);
    //LAPACKE_dlasrt ('I', era.n_offsprings, nullptr );
    std::sort(era.keys_offsprings.data(), era.keys_offsprings.data() + era.n_offsprings,
              [&](std::size_t idx1, std::size_t idx2) { return era.f_offsprings[idx1] < era.f_offsprings[idx2]; });
    for (int i = 0; i < era.n_parents; i++) {
        int idx_new = era.keys_offsprings[i];
        std::copy(era.params_offsprings.memptr(idx_new), era.params_offsprings.memptr(idx_new) + era.n_params,
                  era.params_parents_ranked.memptr(i));
    }
    for (int i = 0; i < era.n_offsprings; i++) {
        int idx_new = era.keys_offsprings[i];
        std::copy(era.y_offsprings.memptr(idx_new), era.y_offsprings.memptr(idx_new) + era.n_params,
                  era.y_offsprings_ranked.memptr(i));
    }
    // <-
}

void CMAES::update_best() {
    double f_cand = cost_function(era.params_parents_ranked.memptr(0));
    if (!std::isnan(f_cand) && f_cand < f_best) {
        std::copy(era.params_parents_ranked.memptr(0), era.params_parents_ranked.memptr(0) + era.n_params,
                  params_best.begin());
        f_best = f_cand;
    }
}

void CMAES::assign_new_mean() {
    era.params_mean_old = era.params_mean;
    std::fill(era.y_mean.begin(), era.y_mean.end(), 0.0);
    MathKernels::mean_vector(era.y_offsprings_ranked.memptr(), era.n_params, era.n_parents, era.n_params, era.w.data(),
                             era.y_mean.data());
    MathKernels::daxpy(era.y_mean.data(), era.params_mean.data(), era.sigma, era.n_params);
}

void CMAES::cummulative_stepsize_adaption() {
    // -> p sigma
    MathKernels::dgemv(era.C_invsqrt.memptr(), 0, era.y_mean.data(), era.p_s.data(), era.n_params, era.n_params,
                       era.n_params, era.p_s_fact, 1.0 - era.c_s);
    // <-
    // -> h sigma
    double p_s_norm = MathKernels::dnrm2(era.n_params, era.p_s.data());
    double p_s_thresh = (1.4 + 2.0 / (era.n_params + 1))
                        * std::sqrt(1.0 - std::pow(1.0 - era.c_s, 2.0 * (era.i_iteration + 1))) * era.chi;
    era.h_sig = p_s_norm < p_s_thresh;
    // <-

    // -> p cov
    MathKernels::dax(era.p_c.data(), era.p_c.data(), 1.0 - era.c_c, era.n_params);
    MathKernels::daxpy(era.y_mean.data(), era.p_c.data(), era.h_sig * era.p_c_fact, era.n_params);
    // <-
}

void CMAES::update_weights() {
    double c_invsqrt_y[n_params * n_params] __attribute__((aligned(32)));
    for (int i = 0; i < era.n_offsprings; i++) {
        if (era.w[i] < 0) {
            MathKernels::dgemv(era.C_invsqrt.memptr(), 0, era.y_offsprings_ranked.memptr(i), c_invsqrt_y,
                               era.n_params, era.n_params, era.n_params, 1.0, 0.0);
            double h = MathKernels::dnrm2(era.n_params, c_invsqrt_y);
            era.w_var[i] = era.w[i] * era.n_params / (h * h);
        }
    }
}

void CMAES::update_cov_matrix() {
    double h1 = (1 - era.h_sig) * era.c_c * (2.0 - era.c_c);
    double h2[n_params * n_params] __attribute__((aligned(32)));
    double h3[n_params * n_params] __attribute__((aligned(32)));
    for (int i = 0; i < n_params * n_params; i++) {
        h2[i] = 0;
        h3[i] = 0;
    }
    for (int i = 0; i < era.n_offsprings; i++) {
        MathKernels::dger(h2, era.y_offsprings_ranked.memptr(i), era.y_offsprings_ranked.memptr(i),
                          era.c_mu * era.w_var[i], era.n_params, era.n_params, era.n_params);
    }
    MathKernels::dger(h3, era.p_c.data(), era.p_c.data(),
                      era.c_1, era.n_params, era.n_params, era.n_params);
    double w_sum = std::accumulate(era.w.data(), era.w.data() + era.n_offsprings, 0.0);
    MathKernels::dgema(era.C.memptr(), era.n_params, era.n_params, era.n_params,
                       (1.0 + era.c_1 * h1 - era.c_1 - era.c_mu * w_sum));
    MathKernels::dgempm(era.C.memptr(), h3, era.n_params, era.n_params, era.n_params);
    MathKernels::dgempm(era.C.memptr(), h2, era.n_params, era.n_params, era.n_params);
}

void CMAES::update_sigma() {
    era.sigma *= std::exp(era.c_s / era.d_s * (MathKernels::dnrm2(era.n_params, era.p_s.data()) / era.chi - 1.0));
}

void CMAES::eigendecomposition() {
    double eigvals_C_sq[n_params] __attribute__((aligned(32)));
    double C_invsqrt_tmp[n_params * n_params] __attribute__((aligned(32)));
    double D_inv[n_params * n_params] __attribute__((aligned(32)));

    std::copy(era.C.memptr(), era.C.memptr() + era.n_params * era.n_params, era.B.memptr());
    MathKernels::dsyevd(era.B.memptr(), era.eigvals_C.data(), era.n_params, era.n_params, era.n_params);
    MathKernels::vdsqrt(era.n_params, era.eigvals_C.data(), eigvals_C_sq);
    MathKernels::diagmat(era.D.memptr(), era.n_params, era.n_params, eigvals_C_sq);
    MathKernels::vdinv(era.n_params, eigvals_C_sq, eigvals_C_sq);
    MathKernels::diagmat(D_inv, era.n_params, era.n_params, eigvals_C_sq);

    MathKernels::dgemm(era.B.memptr(), 0, D_inv, 0, C_invsqrt_tmp, era.n_params, era.n_params,
                       era.n_params, 1.0, 0, era.n_params, era.n_params, era.n_params);
    MathKernels::dgemm(C_invsqrt_tmp, 0, era.B.memptr(), 1, era.C_invsqrt.memptr(), era.n_params,
                       era.n_params, era.n_params, 1.0, 0, era.n_params, era.n_params, era.n_params);
}

void CMAES::stopping_criteria() {
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
    double sigma_up_thresh = 1e20 * std::sqrt(eigval_max);
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
        double ei = 0.1 * era.sigma * era.eigvals_C[i];
        for (int j = 0; j < era.n_params; j++) {
            nea += era.params_mean[i] == era.params_mean[i] + ei * era.B(j, i);
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
        nec += era.params_mean[i] == era.params_mean[i] + 0.2 * era.sigma * std::sqrt(era.C(i, i));
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

double CMAES::cost_function(double *params) {
    transform_scale_shift(params, params_typical.data(), era.params_tss.data(), n_params);
    model->evaluate(data->x, era.params_tss);
    double cost = 0.0;
    for (int i = 0; i < model->dim; i++) {
        cost += MathKernels::least_squares(model->y.memptr(i), data->y.memptr(i), data->n_data);
    }
    return cost;
}

void CMAES::plot(dvec &params) {
#ifndef NDEBUG
    std::cout << "f0: " << std::setprecision(std::numeric_limits<double>::digits10 + 1) << era.f_offsprings[0]
              << std::endl;
#endif
}

dvec CMAES::fmin(dvec &params0_, double sigma0_, dvec &params_typical_, int n_restarts, int seed, tss_type tss_func) {
    // -> settings
    transform_scale_shift = tss_func;
    MathKernels::init_random_number_generator(&rnd_stream, seed);
    n_params = model->n_params;
    i_run = 0;
    // <-

    // -> first guess
    params0 = params0_;
    params_typical = params_typical_;
    sigma0 = sigma0_;
    // <-

    // -> prepare fmin
    n_offsprings0 = (int) (4 + 3 * std::log(n_params));
    n_offsprings = n_offsprings0;
    int n_offsprings_max = n_offsprings0 * (1 << n_restarts);
    era.reserve(n_offsprings_max, n_params);
    era.reinit(n_offsprings, n_params, params0, sigma0);
    params_best = params0;
    double f_cand = cost_function(params_best.data());
    f_best = std::isnan(f_cand) ? std::numeric_limits<double>::infinity() : f_cand;
    i_func_eval_tot = 0;
    int budget[2] = {0, 0};
    // <-

    // -> first run

    Timer::start();
    optimize();
    Timer::stop();
    std::cout << "timing (ms): " << Timer::get_timing() << std::endl;
    std::cout << "i_iteration: " << era.i_iteration << std::setprecision(std::numeric_limits<double>::digits10 + 1)
              << " f_best: " << f_best << std::endl;
    exit(0);
    budget[0] += era.i_func_eval;
    // <-
    // -> restarts
    should_stop_optimization = false;
    for (i_run = 1; i_run < n_restarts + 1 && !should_stop_optimization; i_run++) {
        int n_regime1 = n_offsprings0 * (1 << i_run);
        while (budget[0] > budget[1] && !should_stop_optimization) {
            double u[2];
            MathKernels::sample_random_vars_uniform(&rnd_stream, 2, u, 0.0, 1.0);
            double ur2 = std::pow(u[0], 2.0);
            int n_offsprings = (int) (n_offsprings0 * std::pow(0.5 * n_regime1 / n_offsprings0, ur2));
            double us = u[1];
            double sigma_regime2 = sigma0 * 2.0 * std::pow(10, -2.0 * us);
            era.reinit(n_offsprings, n_params, params_best, sigma_regime2);
            optimize();
            budget[1] += era.i_func_eval;
        }
        n_offsprings = n_regime1;
        era.reinit(n_offsprings, n_params, params_best, sigma0);
        optimize();
        budget[0] += era.i_func_eval;
        std::cout << "i_run: " << i_run << " / " << n_restarts << " completed. f_best: " << f_best << std::endl;
    }
    // <-

    // -> plot final result
    cost_function(params_best.data());
    plot(params_best);
    dvec params_best_unscaled(n_params);
    transform_scale_shift(params_best.data(), params_typical.data(), params_best_unscaled.data(), n_params);
    std::cout << "f_best: " << f_best << std::endl;
    std::cout << "params:";
    for (int i = 0; i < n_params; i++)
        std::cout << " " << params_best_unscaled[i] << " ";
    std::cout << std::endl;
    return params_best_unscaled;

}

