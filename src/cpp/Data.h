#ifndef LIBCMAES_DATA_H
#define LIBCMAES_DATA_H

#include "Types.h"
#include <string>

struct Data {
    int n_data, dim;
    dvec x; // <- 1-D array
    dmat y; // <- N-D array

    virtual void read_data_from_file(std::string filename) = 0;
};


#endif
