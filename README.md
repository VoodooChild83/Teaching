# Work and solutions associated with my teaching assignments

In this repository you can find the codes and solutions associated with my various teaching assignments. 

## Econometrics 2 - Spring 2014

The master's course completes the econometrics sequence, focusing on: the theory of the asymptotic distribution of M-estimators; maximum-likelihood properties, derivation of estimators, and its asymptotic properties; the theory behind GMM estimation, its asymptotic properties, and robust estimation of the asymptotic variance for plausible inference testing; other forms of instrumented regressions (such as IV and the 2-stage least squares); and an introduction to time-series analysis that serves as a primer for the next course.

### Problem Set 1

The PDF solution and the matlab codes associated with the problem set are included, with generalized routines for calculating the HAC corrected variances (implementation of Newey-West). Includes an introduction to time-series analysis and the consequences of a misspecified model that violates weak exogeneity, as well as some Monte-Carlo exercises to study the behaviour of various estimators from finite to "infinte" sample sizes.

### Problem Set 2

Likewise, a PDF solutions is included expanding on the GMM estimation technique and its properties. It also explores how other estimators (such as IV, Maximum Likelihood) can be re-cast as a GMM estimation when proper orthgonality conditions can be used as moments. Further, the consequences of spurious regressions is studied from a time-series perspective. 

## GSE Tutoring - Econometrics

In this file one can find Jupyter Notebooks implemented with the Matlab kernel containing practice programming exercises and homework solutions for those students who I have tutored in the Macroeconmic Policy and Financial Markets Program at the Barcelona Graduate School of Economics. 

Thee following websites detail how get Matlab working in Jupyter Notebooks and in Python (if desired):

* [Guide to install the Matlab kernel](https://anneurai.net/2015/11/12/matlab-based-ipython-notebooks/)
* [Guide to install Matlab magics for use in Jupyter Notebooks](https://arokem.github.io/python-matlab-bridge/)
* Additionally, you may need to create a [simlink/adjust your .bash_profile] (http://superuser.com/questions/381825/running-matlab-from-mac-osx-terminal) accordingly.
* [Installing Matlab Engine API for Python](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html)

### Solutions to Part 2 of Econometrics Course in the Barcelona GSE's MPFMP Master's Program 

This file contains Jupyter Notebooks detailing the solutions to programming homework assignmens. The .m files are also included for those without the Matlab kernel installed in their Jupyter environment.

### Practice

#### Monte-Carlo Exercise of OLS properties

In this module the student can explore the different properties of OLS with finite and asymptotic sample sizes; and exploration of the role of omited variable bias and correlated regressors; and the problem of endogeneity of the regressor and the 2-Stage Least Squares solution to this problem. 

#### Effects of colinear/correlated regressors on inference testing

Since it is very unlikely that random variables would achieve "perfect" collinearity outside of an introduced collinearity by the econometrician, the exercise focuses on exploring how varying degrees of correlation between regressors affects inference testing. The t-test statistic and F-test statistic are studied under these conditions to elucidate their behavior. 

## Microeconomic Theory for the Models and Methods of Quantitative Economics master's program (UAB and Paris Sorbonne)

A Jupyter Notebook describing the solution of a social planner's problem of a CD utlity function using SymPy.
