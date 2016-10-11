#ifndef CMAES_PARAMETERS_H
#define CMAES_PARAMETERS_H

#include <vector>
#include <cmath>
#include <algorithm>
#include "CMAESTypes.h"

namespace CMAES {
    struct Parameter {

        /* reserves memory for arrays and matrices. */
        void reserve(int n_offsprings_reserve_, int n_params_);

        /* reinitializes variables for new generation. */
        void reinit(int n_offsprings_, int n_params_, dvec &x_mean_, double &sigma_);

        int n_offsprings; // <-- number of offsprings in a generation.
        int n_offsprings_reserve; // <-- size to reserve for containers that grow with number of offsprings.
        int n_parents; // <-- number of parents in a generation.
        int n_parents_reserve; // <-- size to reserve for containers grow with number of offsprings.
        int n_params; // <-- number of parameters = dimensions of problem.
        int i_iteration; // <-- generation/iteration counter.
        int i_func_eval; // <-- function evaluation counter.
        double n_mu_eff; // <-- effective number of parents.
        dmat x_offsprings; // <-- offsprings.
        dmat x_parents_ranked; // <-- ranked parents.
        dmat z_offsprings; // <-- multi-variate normal samples/ mutation matrix.
        dmat y_offsprings; // <-- corresponds to offsprings at zero mean.
        dmat y_offsprings_ranked; // <-- ranked variant of y_offsprings.
        dvec f_offsprings; // <-- cost of offsprings.
        dvec w; // <-- fix weights.
        dvec w_var; // <-- variable weights.
        dvec y_mean; // <-- mean of y_offsprings.
        dvec x_mean;  // <-- mean of x_offsprings.
        dvec x_tss; // <-- transformed scaled shifted offspring (helper).
        dvec x_mean_old; // <-- x_mean of previous generation.
        ivec keys_offsprings; // <-- indices of x_offsprings (helper).
        double c_c; // <-- learning rate for cumulation of the rank-one update.
        double c_s; // <-- learning rate for the cumulation for the step-size control.
        double c_1; // <-- learning rate for the rank-one update of the covariance matrix update.
        double c_mu; // <-- learning rate for the rank-μ update of the covariance matrix update.
        double d_s; // <-- damping parameter for step-size update.
        double chi; // <-- expectation of a euclidean norm of a N(0,I) distributed random vector.
        dvec p_c; // <-- evolution path.
        dvec p_s; // <-- conjugate evolution path.
        double p_c_fact; // <-- pre-calculated factor for p_c (helper).
        double p_s_fact; // <-- pre-calculated factor for p_s (helper).
        double sigma; // <-- step-size.
        dvec eigvals_C; // <-- eigenvalues of matrix C.
        dmat C; // <-- covariance matrix.
        dmat C_invsqrt; // <-- C^(-1/2)
        dmat B; // <-- eigenvectors of matrix C.
        dmat D; // <-- eigendecomposed diagonal matrix of C.
        bool h_sig; // <-- Heaviside-function that stalls p_c if norm(p_c) too large.
    };
}

#endif
