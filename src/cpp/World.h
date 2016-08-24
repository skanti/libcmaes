#ifndef LIBCMAES_DATA_H
#define LIBCMAES_DATA_H

#include "Types.h"
#include <string>

struct World {

    virtual void evaluate(dvec &params, int n_params) = 0;

    virtual double cost_func(dvec &params, dvec &params_typical, int n_params)  = 0;
};


#endif
