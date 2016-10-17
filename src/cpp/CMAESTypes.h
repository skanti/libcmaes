#ifndef CMAES_TYPES_H
#define CMAES_TYPES_H

#include <vector>
#include "AlignedAllocator.h"

typedef std::vector<double, AlignedAllocator<double, 32>> dvec;
typedef std::vector<int, AlignedAllocator<int, 32>> ivec;

template<typename T, int A = 32>
struct Matrix {

    Matrix() : n_rows(0), n_cols(0), n_rows_reserve(0), n_cols_reserve(0), data(0) {};

    ~Matrix() { if (data != NULL) free(data); }

    void reserve(int n_rows_reserve_, int n_cols_reserve_) {
        n_rows_reserve = n_rows_reserve_;
        n_cols_reserve = n_cols_reserve_;
        posix_memalign((void **) &data, A, n_rows_reserve * n_cols_reserve * sizeof(T));
    }


    void resize(int n_rows_, int n_cols_) {
        n_rows = n_rows_;
        n_cols = n_cols_;
    }


    void reserve_and_resize(int n_rows_, int n_cols_) {
        reserve(n_rows_, n_cols_);
        resize(n_rows_, n_cols_);
    }

    void eye() {
        for (int j = 0; j < n_cols; j++) {
            for (int i = 0; i < n_rows; i++) {
                data[j * n_rows_reserve + i] = i == j;
            }
        }
    }

    inline T *memptr(int j = 0) {
        return data + j * n_rows_reserve;
    }

    inline T &operator()(int i, int j) {
        return data[j * n_rows_reserve + i];
    }

    int n_rows, n_cols;
    int n_rows_reserve, n_cols_reserve;
    T *data;
};

struct Solution {
    dvec params;
    double f;
    int i_func_evaluations;
};

typedef Matrix<double> dmat;


#endif
