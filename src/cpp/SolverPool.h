#ifndef LIBCMAES_SOLVERPOOL_H
#define LIBCMAES_SOLVERPOOL_H

#include "mkl.h"

struct SolverPool {
    static void
    transform_scale_shift(double *params, double *params_typical, double a_geno, double b_geno, double a_pheno,
                          double b_pheno, int n, double *params_new);

    static void dot(double *v1, double *v2, int n);

    static void mean_vector(double *a, int n_rows_a, int n_cols_a, int ld_a, double *v, double *w);

    static double least_squares(double *v1, double *v2, int n);

    static void logspace(double *v, double a, double b, int n);

    static void
    dgemm(double *a, double *b, double *c, int n_rows_a, int n_cols_a, int n_cols_b, int lda, int ldb, int ldc);

    static void
    dgemv(double *a, double *x, double *y, int n_rows_a, int n_cols_a, int lda, double alpha, double beta);

    static void daxpy(double *x, double *y, double a, int n);

    static double dnrm2(int n, double *x);

    static void vdmul(double *x, double *y, double *z, int n);
};


#endif //LIBCMAES_SOLVERPOOL_H
