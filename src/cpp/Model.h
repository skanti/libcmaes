#ifndef LIBCMAES_MODEL_H
#define LIBCMAES_MODEL_H

#include "Types.h"

struct Model {
    Model(int n_data_, int dim_, int n_params_) : n_data(n_data_), dim(dim_), n_params(n_params_),
                                                  y(dim, dvec(n_data)) { };

    virtual void evaluate(dvec &x, dvec &params) = 0;


    int n_data, dim, n_params;
    std::vector<dvec> y;
};

#endif //LIBCMAES_MODEL_H
