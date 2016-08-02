# libcmaes

A slim and fast C++ BIPOP-aCMA-ES library. (Intel MKL usage).

## Description

This is a C++11 implementation of the BIPOP-aCMA-ES algorithm. The algorithm is referenced 
from "The CMA Evolution Strategy: A Tutorial, Nikolaus Hansen, 2016". Apparently, he has a new setting for the default 
negative weights (since 2016).

## Purpose

You ask yourself why did I bother to create this library, given the fact that there are already a couple of existing
CMA-ES algorithms floating in the internet. The reason is that the source code of mine is ridiculously simple, structured
and straight forward to understand. The compile time is much faster than other templated CMA-ES libraries.
Consider this for academic purposes. You can manipulate/modify the source code and run your own flavor.

It takes literally 5 minutes to go through the code and understand the linear computational path. 
Other templated libraries require hours (if not days) to really unravel the computational path.

It is possible to watch the convergence online. That is, the cost-function is plotted via gnuplot interval-wise.
I found this very partical.

## How-To install

Just have a look at the CMake file. Just run it with your individual paths.

## Dependencies
- Intel MKL
