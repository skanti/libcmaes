#ifndef LIBCMAES_TYPES_H
#define LIBCMAES_TYPES_H

#include "AlignedAllocator.h"

enum StorageType {
    column_major = 0,
    row_major = 1
};

template<typename T>
struct Matrix : public T {
    Matrix(int n_rows_, int n_cols_) : n_rows(n_rows_), n_cols(n_cols_), T(n_rows * n_cols) {};

    Matrix() : n_rows(0), n_cols(0), T(0) {};

    void resize(int n_rows_, int n_cols_) {
        n_rows = n_rows_;
        n_cols = n_cols_;
        T::resize(n_rows * n_cols);
    }

    void eye() {
        for (int j = 0; j < n_cols; j++) {
            for (int i = 0; i < n_rows; i++) {
                this->data[j * n_rows + i] = i == j;
            }
        }
    }

    double *get_col(int j) {
        return this->data() + j * n_rows;
    }

    double &operator()(int i, int j) {
        return this->operator[](j * n_rows + i);
    }

    int n_rows, n_cols;
};

typedef std::vector<double, AlignedAllocator<double, 32>> dvec;
typedef std::vector<int, AlignedAllocator<int, 32>> ivec;
typedef Matrix<dvec> dmat;
#endif
