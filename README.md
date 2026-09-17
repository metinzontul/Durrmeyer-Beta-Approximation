# Durrmeyer-Beta Non-Parametric Regression

**A robust, non-parametric regression estimator and approximation framework based on the Durrmeyer-Beta operator.**

This repository contains the official Python implementation and reproducible benchmarking suite for the Durrmeyer-Beta operator. The project bridges pure approximation theory with applied machine learning by deploying a variation-diminishing mathematical operator as a parameter-free regression estimator capable of outperforming traditional time-series models (ARIMA) and modern deep learning architectures (LSTM Autoencoders) under extreme industrial noise.

## 🚀 Key Features

*   **Non-Parametric Robustness:** Reconstructs underlying continuous manifolds from heavily noisy data without hyperparameter tuning, backpropagation, or weight optimization.
*   **Extreme Noise Tolerance:** Maintains topological integrity and high $R^2$ scores even when the signal-to-noise ratio collapses (tested up to Std = 16.0).
*   **Numerical Stability:** Resolves standard underflow issues in infinite summation operators via a dynamic outer series truncation methodology (omitted mass $< 10^{-14}$) and highly calibrated adaptive quadrature.
*   **Comprehensive Benchmarking:** Includes a zero-shot LSTM Autoencoder and an ARIMA(2,0,2) baseline for direct comparison on synthetic spectroscopic datasets.
*   **Bivariate Extension:** Provides full tensor-product implementations for 3D surface approximations of both smooth and non-smooth functions.

## 🧮 Mathematical Formulation

The proposed univariate Durrmeyer-type operator utilizes Lupaş-Agratini weights and a normalized beta-prime integral kernel:

$$D_n(f; z) = \sum_{k=0}^{\infty} p_{n,k}(z) \int_0^\infty b_{n,k}(t) f(t) dt$$

where the outer weights follow a negative binomial distribution:
$$p_{n,k}(z) = \binom{nz+k-1}{k} \frac{1}{2^{nz+k}}$$

and the inner kernel represents a Beta-prime expectation. To compute this on bounded domains without overflow, the algorithm applies the transformation $x = \frac{t}{1+t}$ mapping the inner integral to the standard Beta distribution $(0,1)$.

## ⚙️ Installation & Requirements

Ensure you have Python 3.9+ installed. Install the required dependencies using `pip`:

```bash
pip install numpy scipy matplotlib scikit-learn statsmodels torch joblib