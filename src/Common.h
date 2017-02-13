#pragma once

#include <eigen3/Eigen/Dense>
#include <random>
#include <iostream>

template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;


typedef Eigen::Matrix<double, Eigen::Dynamic, 1> dvec;
typedef Eigen::Matrix<int32_t, Eigen::Dynamic, 1> ivec;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> dmat;

/*
namespace {
    struct scalar_normal_dist_op {
        static std::mt19937 mt;
        mutable std::normal_distribution<double> dist_normal; 
        
        EIGEN_EMPTY_STRUCT_CTOR(scalar_normal_dist_op)

        template<typename Index>
        inline double operator() (Index, Index = 0) const { return dist_normal(mt); }
        inline void seed(const uint64_t &s) { mt.seed(s); }
    };

    std::mt19937 scalar_normal_dist_op::mt;
      
    struct functor_traits{ 
        enum { 
            Cost = 50 * Eigen::NumTraits<double>::MulCost, PacketAccess = false, IsRepeatable = false 
        }; 
    };

} 


class EigenMultivariateNormal {
    dmat covar;
    dmat transform;
    dvec mean;
    scalar_normal_dist_op randN; // Gaussian functor
    
  public:
    void init(const dvec& mean_, const dmat& covar_, const uint64_t &seed) {
        randN.seed(seed);
        mean = mean_;
        covar = covar_;

        Eigen::LLT<dmat> cholSolver(covar_);
        transform = cholSolver.matrixL();
    }

    Eigen::Matrix<double, Eigen::Dynamic,-1> samples(int nn) {
        return (transform * Eigen::Matrix<double, Eigen::Dynamic,-1>::NullaryExpr(covar.rows(), nn, randN)).colwise() + mean;
    }
};
*/

struct Solution {
    dvec params;
    double f;
};

