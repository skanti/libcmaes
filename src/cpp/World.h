#ifndef LIBCMAES_DATA_H
#define LIBCMAES_DATA_H

#include "CMAESTypes.h"

struct World {

    virtual void evaluate(dvec &params, int n_params) = 0;

    virtual double cost_func(dvec &params, dvec &params_typical, int n_params)  = 0;

    virtual void transform_scale_shift(double *x, double *x_typical, double *x_tss, int n_params) {
        std::swap(x, x_tss);
    }
};


#endif
