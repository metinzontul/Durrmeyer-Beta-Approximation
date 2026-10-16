# -*- coding: utf-8 -*-
"""
Script: bounded_wide_example.py
Author: Metin Zontul
Date:   2026 October
Description: Evaluates the Durrmeyer-Beta operator on a wider interval [0, 20] 
             using a bounded, uniformly continuous target function with a 
             discontinuous second derivative. Includes independent tail integral checks.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln
from scipy.stats import nbinom
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def target_function(t):
    """
    A bounded, uniformly continuous test function (S-curve) on [0, 20].
    Possesses a continuous first derivative but a discontinuous second derivative at t=10.
    """
    u = np.asarray(t) - 10.0
    return 0.5 * (1 + u * np.abs(u) / (1 + u * u))

def expectation(n, k):
    """Calculates the integral expectation for the Durrmeyer-Beta operator."""
    a = k + 1.0
    b = float(n)
    mean = a / (a + b)
    sd = np.sqrt(a * b / ((a + b)**2 * (a + b + 1)))
    
    # Adaptive integration points around the mean to handle the sharp peak
    pts = [10/11, mean] + [mean + j * sd for j in [-10, -6, -3, -1, 1, 3, 6, 10]]
    pts = sorted(set(p for p in pts if 0 < p < 1))
    
    lb = betaln(a, b)
    
    def integrand(x):
        return np.exp(k * np.log(x) + (n - 1) * np.log1p(-x) - lb) * target_function(x / (1 - x))
        
    res = quad(integrand, 0, 1, points=pts, epsabs=2e-11, epsrel=2e-11, limit=500, full_output=1)
    
    if len(res) > 3 or res[1] > 3e-11: 
        raise RuntimeError((n, k, res[1], res[3:] if len(res) > 3 else ''))
        
    return res[0]

def independent_tail_check(n, k):
    """Independent mathematical verification for the integrals on [0, 10] and [10, inf]."""
    lb = betaln(k + 1, n)
    
    def integrand(t):
        if t <= 0: return 0.0
        return np.exp(k * np.log(t) - (k + n + 1) * np.log1p(t) - lb) * target_function(t)
        
    res_lower = quad(integrand, 0, 10, epsabs=1e-11, epsrel=1e-11, limit=500)[0]
    res_upper = quad(integrand, 10, np.inf, epsabs=1e-11, epsrel=1e-11, limit=500)[0]
    return res_lower + res_upper

def main():
    root = Path(__file__).resolve().parent
    z = np.linspace(0, 20, 1001)
    truth = target_function(z)
    apps = []
    
    print("Evaluating Durrmeyer-Beta approximation on [0, 20]...")
    for n in [5, 10, 20, 40, 80]:
        K = int(nbinom.isf(1e-14, 20 * n, 0.5))
        while nbinom.sf(K, 20 * n, 0.5) > 1e-14:
            K += 1
            
        vals = np.array([expectation(n, k) for k in range(K + 1)])
        k_arr = np.arange(K + 1)
        
        W = np.zeros((len(z), len(k_arr)))
        W[0, 0] = 1.0
        W[1:] = nbinom.pmf(k_arr[None, :], n * z[1:, None], 0.5)
        
        app = W @ vals
        apps.append(app)
        err = app - truth
        
        print(f"n={n:<2} | K_max={K:<4} | E_inf={np.max(abs(err)):.6f} | RMSE={np.sqrt(np.mean(err**2)):.6f} | MAE={np.mean(abs(err)):.6f}")
        
        # Perform independent check for boundary conditions
        for kk in ([0, n * 10, n * 20] if n in [5, 80] else []):
            ref = independent_tail_check(n, kk)
            print(f" -> Independent check for k={kk:<4}: difference = {abs(ref - vals[kk]):.3e}")

    # Save raw data for reproducibility
    np.savez(root / 'bounded_wide_results.npz', z=z, target=truth, approximations=np.array(apps))
    
    # Generate Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    axes[0].plot(z, truth, 'k', lw=2.5, label='Target')
    
    colors = ['#D55E00', '#E69F00', '#009E73', '#0072B2', '#7B2CBF']
    for n, ap, c in zip([5, 10, 20, 40, 80], apps, colors):
        axes[0].plot(z, ap, color=c, lw=1.8, label=rf'$D_{{{n}}}$')
        axes[1].plot(z, abs(ap - truth), color=c, lw=1.8, label=rf'$n={n}$')
        
    for ax in axes:
        ax.set_xlabel('$z$', fontsize=12)
        ax.set_xlim(0, 20)
        ax.grid(alpha=0.2)
        
    axes[0].set_ylabel('Function value', fontsize=12)
    axes[1].set_ylabel('Absolute error', fontsize=12)
    axes[0].legend(fontsize=10, ncol=2)
    axes[1].legend(fontsize=10)
    
    output_file = root / 'figure_wider_interval_bounded.png'
    fig.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"Plot saved successfully to: {output_file.name}")

if __name__ == "__main__":
    main()