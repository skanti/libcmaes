#ifndef LIBCMAES_SOLVERPOOL_H
#define LIBCMAES_SOLVERPOOL_H

#include <cmath>

struct SolverPool {
    static void
    transform_scale_shift(double *params, double a_geno, double b_geno, double a_pheno, double b_pheno, int n,
                          double *params_new) {
        double d_pheno = b_pheno - a_pheno;
        double d_geno = b_geno - a_geno;
        for (int i = 0; i < n; i++) {
            params_new[i] = a_pheno + d_pheno / d_geno * params[i];
        }
    }

    static void dot(double *v1, double *v2, int n) {
        double r = 0;
        for (int i = 0; i < n; i++) {
            r += v1[i] * v2[i];
        }
    }

    static void least_squares(double *v1, double *v2, int n) {
        double r = 0;
        for (int i = 0; i < n; i++) {

            double e = v1[i] * v2[i];
            r += e * e;
        }
    }
};


#endif //LIBCMAES_SOLVERPOOL_H
