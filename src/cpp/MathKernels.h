#ifndef LIBCMAES_SOLVERPOOL_H
#define LIBCMAES_SOLVERPOOL_H

#include "mkl.h"

struct MathKernels {

    static void dot(double *v1, double *v2, int n);


    static void init_random_number_generator(VSLStreamStatePtr *stream, const unsigned int seed);

    static void sample_random_vars_gaussian(VSLStreamStatePtr *stream, int n, double *f, double a, double b);

    static void sample_random_vars_uniform(VSLStreamStatePtr *stream, int n, double *f, double a, double b);

    static void mean_vector(double *a, int n_rows_a, int n_cols_a, int ld_a, double *v, double *w);

    static double least_squares(double *v1, double *v2, int n);

    static void logspace(double *v, double a, double b, int n);

    static void dgemm(double *a, int is_a_trans, double *b, int is_b_trans, double *c, int m, int n, int k,
                      double alpha, double beta, int lda, int ldb, int ldc);

    static void dgemv(double *a, int is_a_trans, double *x, double *y, int n_rows_a, int n_cols_a, int lda,
                      double alpha, double beta);


    static void dgemv_c(double *a, double *b, double *x, int n_rows_a, int n_cols_a, int ld_a, int ld_b, double alpha);

    static void dgema(double *a, int n_rows_a, int n_cols_a, int ld_a, double alpha);

    static void dgempm(double *a, double *b, int n_rows_a, int n_cols_a, int ld_a);

    static void dsyevd(double *a, double *w, int n_rows_a, int n_cols_a, int ld_a);

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

    static void dax(double *x, double *y, double a, int n);

    static void dger(double *a, double *x, double *y, double alpha, int n_rows_a, int n_cols_a, int ld_a);

    static double dnrm2(int n, double *x);

    static void vdmul(double *x, double *y, double *z, int n);
};


#endif //LIBCMAES_SOLVERPOOL_H
