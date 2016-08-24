#ifndef LIBCMAES_MODEL_H
#define LIBCMAES_MODEL_H

#include "Types.h"

struct Model {
    Model(int n_data_, int dim_)
            : n_model(n_data_),
              dim(dim_) { y.reserve_and_resize(n_model, dim); };

    virtual void evaluate(dvec &x, dvec &params) = 0;


    int n_model, dim, n_params;
    dmat y;
};

#endif //LIBCMAES_MODEL_H
