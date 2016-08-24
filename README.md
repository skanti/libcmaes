# libcmaes

A slim and aggressively optimized C++ BIPOP-aCMA-ES optimization library. 

I claim that this is (by far) the fastest CMA-ES optimization algorithm available in the internet.

## Description

This is a C++11/Intel MKL implementation of the BIPOP-aCMA-ES algorithm. The algorithm is referenced 
from "The CMA Evolution Strategy: A Tutorial, Nikolaus Hansen, 2016". Apparently, he has a new setting for the default 
negative weights (since 2016).

## Purpose

You ask yourself why did I bother to create this library, given the fact that there are already a couple of existing
CMA-ES algorithms floating in the internet. The reason is that the source code of mine is ridiculously simple, structured
and straight forward to understand. The compile time is much faster than other templated CMA-ES libraries.
Consider this for academic purposes. You can manipulate/modify the source code and run your own flavor.

It takes literally 5 minutes to go through the code and understand the linear computational path. 
Other templated libraries require hours (if not days) to really unravel the computational path.

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
Have a look at the CMake file. Just run it with your individual paths.

## Dependencies
- Intel MKL

## Contact
Feel free to contact me if you have questions or just want to chat about it.
