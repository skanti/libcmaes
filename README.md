# libcmaes

A slim and Intel-MKL optimized C++ BIPOP-aCMA-ES optimization library. 

## Description

This is a C++11/Intel MKL implementation of the BIPOP-aCMA-ES algorithm. The algorithm is referenced 
from "The CMA Evolution Strategy: A Tutorial, Nikolaus Hansen, 2016". Apparently, he has a new setting for the default 
negative weights (since 2016).

## Purpose

The compile time is much faster than other templated CMA-ES libraries. Consider this for academic purposes. You can manipulate/modify the source code and run your own flavor. Also the source code is rather easy to understand because it follows a linear execution path (i.e. not much branching).

It takes 5 minutes to go through the code and understand the linear computational path. 
Other templated libraries require hours (if not days) to really unravel what goes behind the curtain.

## Performance and Optimization
The heavy duty work is mostly done by Intel MKL BLAS level 2 and BLAS level 3. One eigenvalue/eigenvector decomposition for real symmetric matrices is done by Intel MKL LAPACK module ```dsyevd``` - which uses the 'divide-and-conquer' algorithm that computes different results than the standard method but it is faster for larger matrices.

Here is a little info about critical methods in the main thread:

```N```= Number of parameters. (Usually < 200)

```M```= Number of offsprings. (Can grow as much as 2^13)

#### sample offsprings

- O(N\*M): sampling of standard normal random variables via mersenne-twister.
- O(N\*N\*M): dgemm
- O(N^3): dgemm

#### rank and sort

- O(log2 M): sorting **UNOPTIMIZED SO FAR. USE OF STL ```sort```**.

#### cost function
- O(M): evaluation of cost-function. **DANGEROUS BECAUSE USER-PROVIDED**.

#### eigen decomposition
- O(N^3):  divide-and-conquer eigen decomposition.

## TO-DO
- [x] Optimize sampling of random variables.
- [ ] Optimize sorting (with preservation of indices).
- [x] Avoid array resizing by allocating a sufficient reservation memory.


## How-To install
Have a look at the CMake file. Have special care with the linking of the Intel-MKL libraries. You may specify your ```LD_LIBRARY_PATH``` to look for shared libraries.

## Dependencies
- Intel MKL

## Contact
Feel free to contact me if you have questions or just want to chat about it.
