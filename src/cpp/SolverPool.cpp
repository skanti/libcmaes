
#include <cmath>
#include "SolverPool.h"

void
SolverPool::transform_scale_shift(double *params, double *params_typical, double a_geno, double b_geno, double a_pheno,
                                  double b_pheno, int n, double *params_new) {
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

void SolverPool::least_squares(double *v1, double *v2, int n) {
    double r = 0;
    for (int i = 0; i < n; i++) {

        double e = v1[i] * v2[i];
        r += e * e;
    }
}

void SolverPool::logspace(double *v, double a, double b, int n) {
    double d = (b - a) / (n - 1);
    for (int i = 0; i < n; i++) {
        v[i] = std::pow(10, d * i);
    }
}



