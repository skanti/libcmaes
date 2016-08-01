
#include <cmath>
#include "SolverPool.h"

void SolverPool::transform_scale_shift(double *params, double *params_typical, double a_geno,
                                       double b_geno, double a_pheno, double b_pheno, int n, double *params_new) {
    double d_pheno = b_pheno - a_pheno;
    double d_geno = b_geno - a_geno;
    double scale = a_pheno + d_pheno / d_geno;
    for (int i = 0; i < n; i++) {
        params_new[i] = params[i] * params[i] * params_typical[i];
    }
}

void SolverPool::dot(double *v1, double *v2, int n) {
    double r = 0;
    for (int i = 0; i < n; i++) {
        r += v1[i] * v2[i];
    }
}

double SolverPool::least_squares(double *v1, double *v2, int n) {
    double sum_least_squares = 0;
    for (int i = 0; i < n; i++) {
        double e = (v1[i] - v2[i]);
        sum_least_squares += e * e;
    }
    return sum_least_squares;
}

void SolverPool::mean_vector(double *a, int n_rows_a, int n_cols_a, int ld_a, double *v, double *w) {

    for (int j = 0; j < n_cols_a; j++) {
        cblas_daxpy(n_rows_a, v[j], a + j * ld_a, 1, w, 1);
    }
}

void SolverPool::logspace(double *v, double a, double b, int n) {
    double d = (b - a) / (n - 1);
    for (int i = 0; i < n; i++) {
        v[i] = std::pow(10, d * i);
    }
}

void SolverPool::dgemm(double *a, int is_a_trans, double *b, int is_b_trans, double *c, int m, int n, int k,
                       double alpha, double beta, int lda, int ldb, int ldc) {
    // C = alpha*A*B + beat*C
    cblas_dgemm(CblasColMajor,
                is_a_trans ? CblasTrans : CblasNoTrans, is_b_trans ? CblasTrans : CblasNoTrans,
                m, k, n,
                alpha,
                a, lda,
                b, ldb,
                beta,
                c, ldc);
}

void SolverPool::dgemv(double *a, int is_a_trans, double *x, double *y, int n_rows_a, int n_cols_a, int lda,
                       double alpha, double beta) {
    // y = alpha*A*x + beta*y,
    cblas_dgemv(CblasColMajor, is_a_trans ? CblasTrans : CblasNoTrans,
                n_rows_a, n_cols_a,
                alpha,
                a, lda,
                x, 1,
                beta,
                y, 1);
}


void
SolverPool::dgemv_c(double *a, double *b, double *x, int n_rows_a, int n_cols_a, int ld_a, int ld_b, double alpha) {
    // B = alpha * x |* A <- column-wise multiplication
    for (int j = 0; j < n_cols_a; j++) {
        for (int i = 0; i < n_rows_a; i++) {
            b[j * ld_b + i] = alpha * x[i] * a[j * ld_a + i];
        }
    }
}

double SolverPool::dnrm2(int n, double *x) {
    return cblas_dnrm2(n, x, 1);
}

void SolverPool::dger(double *a, double *x, double *y, double alpha, int n_rows_a, int n_cols_a, int ld_a) {
    cblas_dger(CblasColMajor, n_cols_a, n_rows_a,
               alpha,
               x, 1,
               y, 1,
               a, ld_a);
}


void SolverPool::dgema(double *a, int n_rows_a, int n_cols_a, int ld_a, double alpha) {
    for (int j = 0; j < n_cols_a; j++) {
        for (int i = 0; i < n_rows_a; i++) {
            a[j * ld_a + i] *= alpha;
        }
    }
}

void SolverPool::dgempm(double *a, double *b, int n_rows_a, int n_cols_a, int ld_a) {
    for (int j = 0; j < n_cols_a; j++) {
        for (int i = 0; i < n_rows_a; i++) {
            a[j * ld_a + i] += b[j * ld_a + i];
        }
    }
}

void SolverPool::dsyev(double *a, double *w, int n_rows_a, int n_cols_a, int ld_a) {

    LAPACKE_dsyev(LAPACK_COL_MAJOR,
                  'V', 'L',
                  n_rows_a,
                  a, ld_a,
                  w);
}

void SolverPool::vdsqrtinv(int n, double *a, double *y) {
    for (int i = 0; i < n; i++) {
        y[i] = 1.0 / std::sqrt(a[i]);
    }
}

void SolverPool::vdsqrt(int n, double *a, double *y) {
    // y = sqrt(a)
    vdSqrt(n, a, y);
}

void SolverPool::vdinv(int n, double *a, double *y) {
    // y = inv(a)
    vdInv(n, a, y);
}

void SolverPool::daxpy(double *x, double *y, double a, int n) {
    cblas_daxpy(n,
                a,
                x, 1,
                y, 1);
}


void SolverPool::vdmul(double *x, double *y, double *z, int n) {
    for (int i = 0; i < n; i++) {
        z[i] = x[i] * y[i];
    }
}

