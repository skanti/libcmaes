#include "Parameters.h"

void Parameters::init(int n_offsprings_, int n_params_, dvec &params_mean_, double &sigma_) {
    n_offsprings = n_offsprings_;
    n_parents = n_offsprings / 2;
    n_params = n_params_;
    i_iteration = 0;
    i_func_eval = 0;

    //-> weights tmp
    dvec w_tmp(n_offsprings);
    double w_neg_sum = 0.0, w_pos_sum = 0.0;
    for (int i = 0; i < n_offsprings; i++) {
        w_tmp[i] = std::log((n_offsprings + 1.0) / 2.0) - std::log(i + 1);
        if (w_tmp[i] >= 0)
            w_pos_sum += w_tmp[i];
        else
            w_neg_sum += w_tmp[i];
    }
    // <-

    // -> vectors
    f_offsprings.resize(n_offsprings);
    p_s.resize(n_params);
    p_s.zeros();
    p_c.resize(n_params);
    p_c.zeros();
    params_mean.resize(n_params);
    params_mean = params_mean_;
    params_mean_old.resize(n_params);
    y_mean.resize(n_params);
    keys_offsprings.resize(n_offsprings);
    // <-

    //-> vector of vectors
    params_offsprings.resize(n_offsprings);
    y_offsprings.resize(n_offsprings);
    y_offsprings_ranked.resize(n_offsprings);
    z_offsprings.resize(n_offsprings);
    for (int i = 0; i < n_offsprings; i++) {
        params_offsprings[i].resize(n_params);
        y_offsprings[i].resize(n_params);
        y_offsprings_ranked[i].resize(n_params);
        z_offsprings[i].resize(n_params);
    }

    params_parents_ranked.resize(n_parents);
    y_parents_ranked.resize(n_parents);
    for (int i = 0; i < n_parents; i++) {
        params_parents_ranked[i].resize(n_params);
        y_parents_ranked[i].resize(n_params);
    }
    // <-

    double w_sum_parent = 0.0, w_sq_sum_parent = 0.0;
    for (int i = 0; i < n_parents; i++) {
        w_sum_parent += w_tmp[i];
        w_sq_sum_parent += w_tmp[i] * w_tmp[i];
    }
    n_mu_eff = w_sum_parent * w_sum_parent / w_sq_sum_parent;

    double w_sum_neg = 0.0, w_sq_sum_neg = 0.0;
    for (int i = n_parents; i < n_offsprings; i++) {
        w_sum_neg += w_tmp[i];
        w_sq_sum_neg += w_tmp[i] * w_tmp[i];
    }

    c_m = 1.0;
    c_s = (n_mu_eff + 2.0) / (n_params + n_mu_eff + 5.0);
    c_c = (4.0 + n_mu_eff / n_params) / (n_params + 4.0 + 2.0 * n_mu_eff / n_params);
    c_1 = 2.0 / (std::pow((n_params + 1.3), 2) + n_mu_eff);
    c_mu = 2.0 * (n_mu_eff - 2.0 + 1.0 / n_mu_eff) / (std::pow(n_params + 2.0, 2) + n_mu_eff);
    c_mu = std::min(1.0 - c_1, c_mu);

    d_s = 1.0 + c_s + 2.0 * std::max(0.0, std::sqrt((n_mu_eff - 1) / (n_params + 1)) - 1);

    chi = std::sqrt(n_params) * (1.0 - 1.0 / (4.0 * n_params) + 1.0 / (21.0 * n_params * n_params));

    // constants used in covariance update.
    p_s_fact = std::sqrt(c_s * (2.0 - c_s) * n_mu_eff);
    p_c_fact = std::sqrt(c_c * (2.0 - c_c) * n_mu_eff);

    // -> parameter for active cma-es
    a_mu = 1.0 + c_1 / c_mu;
    a_mueff = 1.0 + 2 * n_mu_eff;
    a_posdef = (1.0 - c_1 - c_mu) / (n_params * c_mu);
    // <-

    // -> weights
    double a_min = std::min(std::min(a_mu, a_mueff), a_posdef);
    w.resize(n_offsprings);
    w_var.resize(n_offsprings);
    for (int i = 0; i < n_offsprings; i++) {
        if (w_tmp[i] >= 0)
            w[i] = w_tmp[i] / w_pos_sum;
        else
            w[i] = a_min * w_tmp[i] / std::abs(w_neg_sum);
    }
    w_var = w;
    // <-

    sigma = sigma_;

    // -> matrices
    C.resize(n_params, n_params);
    C_invsqrt.resize(n_params, n_params);
    B.resize(n_params, n_params);
    D.resize(n_params, n_params);
    C.eye();
    C_invsqrt.eye();
    B.eye();
    D.eye();
    // <-

}
