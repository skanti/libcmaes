
#include <cmath>
#include "MathKernels.h"

namespace CMAES {


    void MathKernels::sample_random_vars_uniform(VSLStreamStatePtr *stream, int n, double *f, double a, double b) {
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, *stream, n, f, a, b);
    }

    void MathKernels::sample_random_vars_gaussian(VSLStreamStatePtr *stream, int n, double *f, double a, double b) {
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, *stream, n, f, a, b);
    }

    void MathKernels::init_random_number_generator(VSLStreamStatePtr *stream, const unsigned int seed) {
        vslNewStream(stream, VSL_BRNG_MT19937, seed);
    }

    void MathKernels::mean_vector(double *a, int n_rows_a, int n_cols_a, int ld_a, double *v, double *w) {

        for (int j = 0; j < n_cols_a; j++) {
            cblas_daxpy(n_rows_a, v[j], a + j * ld_a, 1, w, 1);
        }
    }


    void MathKernels::dgemm(double *a, int is_a_trans, double *b, int is_b_trans, double *c, int m, int n, int k,
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

    void MathKernels::dgemv(double *a, int is_a_trans, double *x, double *y, int n_rows_a, int n_cols_a, int lda,
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
    MathKernels::dgemv_c(double *a, double *b, double *x, int n_rows_a, int n_cols_a, int ld_a, int ld_b,
                         double alpha) {
        // B = alpha * x |* A <- column-wise multiplication
        for (int j = 0; j < n_cols_a; j++) {
            for (int i = 0; i < n_rows_a; i++) {
                b[j * ld_b + i] = alpha * x[i] * a[j * ld_a + i];
            }
        }
    }

    double MathKernels::dnrm2(int n, double *x) {
        return cblas_dnrm2(n, x, 1);
    }

    void MathKernels::dger(double *a, double *x, double *y, double alpha, int n_rows_a, int n_cols_a, int ld_a) {
        cblas_dger(CblasColMajor, n_cols_a, n_rows_a,
                   alpha,
                   x, 1,
                   y, 1,
                   a, ld_a);
    }


    void MathKernels::dgema(double *a, int n_rows_a, int n_cols_a, int ld_a, double alpha) {
        for (int j = 0; j < n_cols_a; j++) {
            for (int i = 0; i < n_rows_a; i++) {
                a[j * ld_a + i] *= alpha;
            }
        }
    }

    void MathKernels::dgempm(double *a, double *b, int n_rows_a, int n_cols_a, int ld_a) {
        for (int j = 0; j < n_cols_a; j++) {
            for (int i = 0; i < n_rows_a; i++) {
                a[j * ld_a + i] += b[j * ld_a + i];
            }
        }
    }

    void MathKernels::dsyevd(double *a, double *w, int n_rows_a, int n_cols_a, int ld_a) {

        LAPACKE_dsyevd(LAPACK_COL_MAJOR,
                       'V', 'L',
                       n_rows_a,
                       a, ld_a,
                       w);
    }

    void MathKernels::vdsqrt(int n, double *a, double *y) {
        // y = sqrt(a)
        vdSqrt(n, a, y);
    }

    void MathKernels::vdinv(int n, double *a, double *y) {
        // y = inv(a)
        vdInv(n, a, y);
    }

    void MathKernels::daxpy(double *x, double *y, double a, int n) {
        // y = a*x + y
        cblas_daxpy(n,
                    a,
                    x, 1,
                    y, 1);
    }

    void MathKernels::dax(double *x, double *y, double a, int n) {
        // y = a*x
        for (int i = 0; i < n; i++) {
            y[i] = a * x[i];
        }
    }


    void MathKernels::vdmul(double *x, double *y, double *z, int n) {
        for (int i = 0; i < n; i++) {
            z[i] = x[i] * y[i];
        }
    }

}