# Durrmeyer-Beta Non-Parametric Regression

**A robust, non-parametric regression estimator and approximation framework based on the Durrmeyer-Beta operator.**

This repository contains the official Python implementation and reproducible benchmarking suite for the Durrmeyer-Beta operator. The project bridges pure approximation theory with applied machine learning by deploying a variation-diminishing mathematical operator as a parameter-free regression estimator. It demonstrates superior structural integrity and outperformance against traditional time-series models (ARIMA) and modern deep learning architectures (LSTM Autoencoders) under extreme industrial noise.

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

To compute the Beta-prime expectation on bounded domains without numerical overflow, the algorithm applies the transformation $x = \frac{t}{1+t}$, mapping the inner integral to the standard Beta distribution over $(0,1)$.

## 📁 Repository Structure & Python Scripts

The codebase is strictly organized into performance benchmarking (signal processing) and theoretical approximation evaluations.

### 📊 1. Benchmarking & Machine Learning (Author: Metin Zontul)
These scripts evaluate the operator's capabilities as a non-parametric regression tool under extreme Gaussian noise.
*   `dbsl_vs_arima_lstm_benchmark.py`: The core evaluation engine. Executes a parallelized comparative robustness analysis evaluating the Durrmeyer-Beta operator against an ARIMA(2,0,2) model and a Zero-Shot LSTM Autoencoder across progressive noise levels.
*   `dbsl_benchmark_plot.py`: Visualization module. Consumes the benchmark outputs to generate high-resolution (600 DPI), publication-ready plots illustrating error degradation (NRMSE, NMAE) and structural integrity ($R^2$).

### 📐 2. Approximation Theory & Bivariate Analysis (Author: Seda Karateke)
These scripts handle the theoretical validation, error distribution, and tensor-product extensions of the operator.
*   `generate_univariate_figures.py`: Evaluates the univariate operator on bounded continuous test functions over $[0, 2]$, generating approximation curves and absolute error distribution plots.
*   `generate_bivariate_approximation_figures.py`: Evaluates the tensor-product Durrmeyer-Beta operator for 3D surfaces. Generates colormapped 3D projections and absolute error heatmaps for both smooth and non-smooth targets.
*   `calculate_bivariate_metrics.py`: Computes theoretical bivariate approximation accuracy metrics (RMSE, MAE, R², $E_\infty$) and outputs the definitive results to a structured CSV file for manuscript tables.

## ⚙️ Installation & Requirements

Ensure you have Python 3.9+ installed. You can install all necessary dependencies using the provided `requirements.txt` file.

```bash
# Clone the repository
git clone [https://github.com/YourUsername/Durrmeyer-Beta-Regression.git](https://github.com/YourUsername/Durrmeyer-Beta-Regression.git)
cd Durrmeyer-Beta-Regression

# Install required packages
pip install -r requirements.txt