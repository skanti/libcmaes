#ifndef LIBCMAES_DATA_H
#define LIBCMAES_DATA_H

#include "Types.h"
#include <string>

struct Data {
    int n_data, dim;
    dvec x; // <- 1-D array
    dmat y; // <- N-D array
};


#endif
