#ifndef LIBCMAES_DATA_H
#define LIBCMAES_DATA_H

#include "Types.h"
#include <vector>

struct Data {
    int n_data, dim;
    dvec x; // <- 1-D array
    std::vector<dvec> y; // <- N-D array

    virtual void populate() = 0;
};


#endif
