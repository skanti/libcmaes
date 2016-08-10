#ifndef LIBCMAES_TYPES_H
#define LIBCMAES_TYPES_H

#include <vector>
#include "AlignedAllocator.h"

enum StorageType {
    column_major = 0,
    row_major = 1
};


template<typename T, typename A = AlignedAllocator<T, 32>>
struct Matrix {
    Matrix(int n_rows_, int n_cols_)
            : n_rows(n_rows_), n_cols(n_cols_), n_rows_reserve(n_rows), n_cols_reserve(n_cols),
              data(n_rows * n_cols) {};

    Matrix() : n_rows(0), n_cols(0), data(0) {};

    void reserve(int n_rows_reserve_, int n_cols_reserve_) {
        n_rows_reserve = n_rows_reserve_;
        n_cols_reserve = n_cols_reserve_;
        data.resize(n_rows_reserve * n_cols_reserve);
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

    inline T *memptr(int j) {
        return data.data() + j * n_rows_reserve;
    }

    inline T &operator()(int i, int j) {
        return data[j * n_rows_reserve + i];
    }

    inline double *memptr() {
        return data.data();
    }

    int n_rows, n_cols;
    int n_rows_reserve, n_cols_reserve;
    std::vector<T, A> data;
};


typedef std::vector<double, AlignedAllocator<double, 32>> dvec;
typedef std::vector<int, AlignedAllocator<int, 32>> ivec;
typedef Matrix<double> dmat;
#endif
