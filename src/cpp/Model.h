#ifndef LIBCMAES_MODEL_H
#define LIBCMAES_MODEL_H

#include "Types.h"

struct Model {
    Model(int n_data_, int dim_)
            : n_model(n_data_),
              dim(dim_),
              y(dim, dvec(n_model)) { };

    virtual void evaluate(dvec &x, dvec &params) = 0;


    int n_model, dim, n_params;
    std::vector<dvec> y;
};

#endif //LIBCMAES_MODEL_H
