#ifndef LIBCMAES_SOLVERPOOL_H
#define LIBCMAES_SOLVERPOOL_H


struct SolverPool {
    static void
    transform_scale_shift(double *params, double *params_typical, double a_geno, double b_geno, double a_pheno,
                          double b_pheno, int n, double *params_new);

    static void dot(double *v1, double *v2, int n);

    static void least_squares(double *v1, double *v2, int n);

    static void logspace(double *v, double a, double b, int n);
};


#endif //LIBCMAES_SOLVERPOOL_H
