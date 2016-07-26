#ifndef LIBCMAES_TYPES_H
#define LIBCMAES_TYPES_H

#include "AlignedAllocator.h"

enum StorageType {
    column_major = 0,
    row_major = 1
};

template<typename T>
struct Matrix {
    Matrix(int n_rows_, int n_cols_) : n_rows(n_rows_), n_cols(n_cols_), data(n_rows * n_cols) {};

    Matrix() : n_rows(0), n_cols(0), data(0) {};

    void resize(int n_rows_, int n_cols_) {
        n_rows = n_rows_;
        n_cols = n_cols_;
        data.resize(n_rows * n_cols);
    }

    double &operator()(int i, int j) {
        return data[j * n_rows + i];
    }

    int n_rows, n_cols;
    T data;
};

typedef std::vector<double, AlignedAllocator<double, 32>> dvec;
typedef std::vector<int, AlignedAllocator<int, 32>> ivec;
typedef Matrix<dvec> dmat;
#endif
