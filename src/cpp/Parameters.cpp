#include "Parameters.h"

void Parameters::init(int n_offsprings_, int n_params_) {
    n_offsprings = n_offsprings_;
    n_parents = n_offsprings / 2;
    n_params = n_params_;
    i_iteration = 0;
    i_evals = 0;
    //-> weights
    w_parents.resize(n_parents);
    double w_sum = 0.0, w_sq = 0.0;
    for (int i = 0; i < n_parents; i++) {
        w_parents[i] = std::log(n_parents + 1) - std::log(i + 1);
        w_sum += w_parents[i];
        w_sq += w_parents[i] * w_parents[i];
    }
    w_parents /= w_sum;
    // <-

    // -> vectors
    f_offsprings.resize(n_offsprings);
    p_s.resize(n_params);
    p_c.resize(n_params);
    params_mean.resize(n_params);
    params_mean_old.resize(n_params);
    y_mean.resize(n_params);
    keys_offsprings.resize(n_offsprings);
    // <-

    //-> vector of vectors
    params_offsprings.resize(n_offsprings);
    y_offsprings.resize(n_offsprings);
    z_offsprings.resize(n_offsprings);
    for (int i = 0; i < n_offsprings; i++) {
        params_offsprings[i].resize(n_params);
        y_offsprings[i].resize(n_params);
        z_offsprings[i].resize(n_params);
    }

    params_parents.resize(n_parents);
    y_parents.resize(n_parents);
    for (int i = 0; i < n_parents; i++) {
        params_parents[i].resize(n_params);
        y_parents[i].resize(n_params);
    }
    // <-

    n_parents_eff = w_sum * w_sum / w_sq;
    c_s = (n_parents_eff + 2.0) / (n_params + n_parents_eff + 5.0);
    c_c = (4.0 + n_parents_eff / n_params) / (n_params + 4.0 + 2.0 * n_parents_eff / n_params);

    c_1 = 2.0 / (std::pow((n_params + 1.3), 2) + n_parents_eff);
    c_mu = 2.0 * (n_parents_eff - 2.0 + 1.0 / n_parents_eff) / (std::pow(n_params + 2.0, 2) + n_parents_eff);
    c_mu = std::min(1.0 - c_1, c_mu);

    d_s = 1.0 + c_s + 2.0 * std::max(0.0, std::sqrt((n_parents_eff - 1) / (n_params + 1)) - 1);

    // constants used in covariance update.
    p_s_fact = std::sqrt(c_s * (2.0 - c_s) * n_parents_eff);
    p_c_fact = std::sqrt(c_c * (2.0 - c_c) * n_parents_eff);

    chi = std::sqrt(n_params) * (1.0 - 1.0 / (4.0 * n_params) + 1.0 / (21.0 * n_params * n_params));

    // -> matrices
    C.resize(n_params, n_params);
    C_sym.resize(n_params, n_params);
    C_invsqrt.resize(n_params, n_params);
    B.resize(n_params, n_params);
    D.resize(n_params, n_params);
    C.eye();
    C_sym.eye();
    C_invsqrt.eye();
    B.eye();
    D.eye();
    // <-

}
