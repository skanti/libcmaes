#include "Parameter.h"
#include <algorithm>
#include <cmath>

namespace CMAES {

    void Parameter::reserve(int n_offsprings_reserve_, int n_parents_reserve_, int n_params_) {
        // N = n_params
        // M = n_offsprings
        // K = n_parents

        n_params = n_params_;
        n_offsprings_reserve = n_offsprings_reserve_;
        n_parents_reserve = n_parents_reserve_;

        // -> N vectors
        p_s.resize(n_params);
        p_c.resize(n_params);
        x_mean.resize(n_params);
        x_mean_old.resize(n_params);
        x_tss.resize(n_params);
        y_mean.resize(n_params);
        eigvals_C.resize(n_params);
        // <-

        // -> M vectors
        f_offsprings.resize(n_offsprings_reserve);
        keys_offsprings.resize(n_offsprings_reserve);
        w.resize(n_offsprings_reserve);
        w_var.resize(n_offsprings_reserve);
        // <-

        //-> NxM matrices
        x_offsprings.resize(n_params, n_offsprings_reserve);
        y_offsprings.resize(n_params, n_offsprings_reserve);
        y_offsprings_ranked.resize(n_params, n_offsprings_reserve);
        z_offsprings.resize(n_params, n_offsprings_reserve);
        // <-

        // -> NxK matrices
        x_parents_ranked.resize(n_params, n_parents_reserve);
        // <-

        // -> NxN matrices
        C.resize(n_params, n_params);
        C_invsqrt.resize(n_params, n_params);
        B.resize(n_params, n_params);
        D.resize(n_params, n_params);
        // <-
    }


    void Parameter::reinit(int n_offsprings_, int n_parents_, int n_params_, dvec &x_mean_, double &sigma_) {
        n_params = n_params_;
        n_offsprings = n_offsprings_;
        n_parents = n_parents_;
        i_iteration = 0;
        i_func_eval = 0;

        x_mean = x_mean_; // <-- setting params mean

        //-> weights tmp
        double w_neg_sum = 0.0, w_pos_sum = 0.0;
        for (int i = 0; i < n_offsprings; i++) {
            w[i] = std::log((n_offsprings + 1.0)/2.0) - std::log(i + 1);
            if (w[i] >= 0)
                w_pos_sum += w[i];
            else
                w_neg_sum += w[i];
        }


        double w_sum_parent = 0.0, w_sq_sum_parent = 0.0;
        for (int i = 0; i < n_parents; i++) {
            w_sum_parent += w[i];
            w_sq_sum_parent += w[i]*w[i];
        }
        n_mu_eff = w_sum_parent*w_sum_parent/w_sq_sum_parent;
        // <-

        // -> N vectors
        std::fill(p_s.data(), p_s.data() + n_params, 0.0);
        std::fill(p_c.data(), p_c.data() + n_params, 0.0);
        // <-

        // -> learning variables
        c_s = (n_mu_eff + 2.0)/(n_params + n_mu_eff + 5.0);
        c_c = (4.0 + n_mu_eff/n_params)/(n_params + 4.0 + 2.0*n_mu_eff/n_params);
        c_1 = 2.0/(std::pow((n_params + 1.3), 2) + n_mu_eff);
        c_mu = 2.0*(n_mu_eff - 2.0 + 1.0/n_mu_eff)/(std::pow(n_params + 2.0, 2) + n_mu_eff);
        c_mu = std::min(1.0 - c_1, c_mu);
        d_s = 1.0 + c_s + 2.0*std::max(0.0, std::sqrt((n_mu_eff - 1)/(n_params + 1)) - 1);
        chi = std::sqrt(n_params)*(1.0 - 1.0/(4.0*n_params) + 1.0/(21.0*n_params*n_params));
        p_s_fact = std::sqrt(c_s*(2.0 - c_s)*n_mu_eff);
        p_c_fact = std::sqrt(c_c*(2.0 - c_c)*n_mu_eff);
        // <-

        // -> active cma-es
        double a_mu = 1.0 + c_1/c_mu;
        double a_mueff = 1.0 + 2*n_mu_eff;
        double a_posdef = (1.0 - c_1 - c_mu)/(n_params*c_mu);
        // <-

        // -> weights
        double a_min = std::min(std::min(a_mu, a_mueff), a_posdef);
        for (int i = 0; i < n_offsprings; i++) {
            w[i] = w[i] >= 0 ? w[i] = w[i]/w_pos_sum : w[i] = a_min*w[i]/std::abs(w_neg_sum);
        }
        w_var = w;
        // <-

        sigma = sigma_; // <- sigma

        // -> NxN matrices
        C.setIdentity();
        C_invsqrt.setIdentity();
        B.setIdentity();
        D.setIdentity();
        // <-

    }
}