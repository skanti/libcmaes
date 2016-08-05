#ifndef LIBCMAES_TYPES_H
#define LIBCMAES_TYPES_H

//#include "AlignedAllocator.h"

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

    void eye() {
        for (int j = 0; j < n_cols; j++) {
            for (int i = 0; i < n_rows; i++) {
                data[j * n_rows + i] = i == j;
            }
        }
    }

    double *get_col(int j) {
        return data.data() + j * n_rows;
    }

    inline double &operator()(int i, int j) {
        return data[j * n_rows + i];
    }

    double *memptr() {
        return data.data();
    }

    int n_rows, n_cols;
    T data;
};

typedef std::vector<double> dvec;
typedef std::vector<dvec> dvecvec;
typedef std::vector<int> ivec;
typedef Matrix<dvec> dmat;
#endif
