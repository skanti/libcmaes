#ifndef LIBCMAES_SOLVERPOOL_H
#define LIBCMAES_SOLVERPOOL_H

#include "mkl.h"

struct SolverPool {
    static void transform_scale_shift(double *params, double *params_typical, double a_geno,
                                      double b_geno, double a_pheno, double b_pheno, int n, double *params_new);

    static void dot(double *v1, double *v2, int n);

    static void mean_vector(double *a, int n_rows_a, int n_cols_a, int ld_a, double *v, double *w);

    static double least_squares(double *v1, double *v2, int n);

    static void logspace(double *v, double a, double b, int n);

    static void
    dgemm(double *a, int is_a_trans, double *b, int is_b_trans, double *c, int m, int n, int k, double alpha,
          double beta, int lda, int ldb, int ldc);

    static void dgemv(double *a, double *x, double *y, int n_rows_a, int n_cols_a, int lda, double alpha, double beta);

    static void dgema(double *a, int n_rows_a, int n_cols_a, int ld_a, double alpha);

    static void dgempm(double *a, double *b, int n_rows_a, int n_cols_a, int ld_a);

    static void dsyev(double *a, double *w, int n_rows_a, int n_cols_a, int ld_a);

    static void vdsqrtinv(int n, double *a, double *y);

    static void vdsqrt(int n, double *a, double *y);

    static void vdinv(int n, double *a, double *y);

    static void diagmat(double *a, int n_rows_a, int ld_a, double *d) {
        for (int j = 0; j < n_rows_a; j++) {
            for (int i = 0; i < n_rows_a; i++) {
                a[j * ld_a + i] = d[j] * (i == j);
            }
        }
    }

    static void daxpy(double *x, double *y, double a, int n);

    static void dger(double *a, double *x, double *y, double alpha, int n_rows_a, int n_cols_a, int ld_a);

    static double dnrm2(int n, double *x);

    static void vdmul(double *x, double *y, double *z, int n);
};


#endif //LIBCMAES_SOLVERPOOL_H
