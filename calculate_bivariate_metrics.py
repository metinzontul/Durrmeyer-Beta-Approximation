# -*- coding: utf-8 -*-
"""
Script: calculate_bivariate_metrics.py
Author: Seda Karateke
Description: Calculates bivariate approximation accuracy metrics (RMSE, MAE, R², E_inf) 
             for the tensor-product Durrmeyer-Beta operator.

Outputs: bivariate_metrics.csv
"""

import csv
from pathlib import Path
import numpy as np
from scipy.special import betainc, gammaln
from scipy.stats import nbinom

# ============================================================
# 1. Numerical settings
# ============================================================
N_VALUES = (5, 10, 20, 40, 80)
Z_MIN, Z_MAX = 0.0, 2.0
GRID_SIZE = 201
TAIL_TOL = 1.0e-14
OUTPUT_DIR = Path(__file__).resolve().parent

# ============================================================
# 2. Test functions
# ============================================================
def phi(z):
    z = np.asarray(z, dtype=float)
    return 4.0 * z / (1.0 + z) ** 2

def psi(z):
    z = np.asarray(z, dtype=float)
    return np.abs(z - 1.0) / (1.0 + z)

def smooth_target(z1, z2):
    return phi(z1) * phi(z2)

def nonsmooth_target(z1, z2):
    return 0.5 * (psi(z1) + psi(z2))

# ============================================================
# 3. Outer Durrmeyer-Beta weights
# ============================================================
def choose_k_max(n):
    r_max = n * Z_MAX
    k_max = int(np.ceil(nbinom.ppf(1.0 - TAIL_TOL, r_max, 0.5)))
    tail = float(nbinom.sf(k_max, r_max, 0.5))
    if tail > 5.0 * TAIL_TOL: raise RuntimeError(f"Series tail is too large for n={n}: {tail:.3e}")
    return k_max, tail

def outer_weights(n, z_grid, k_max):
    k = np.arange(k_max + 1, dtype=float)
    weights = np.zeros((z_grid.size, k_max + 1), dtype=float)
    for row, z in enumerate(z_grid):
        r = n * float(z)
        if r == 0.0:
            weights[row, 0] = 1.0
            continue
        log_weights = -r * np.log(2.0) + gammaln(r + k) - gammaln(r) - gammaln(k + 1.0) - k * np.log(2.0)
        weights[row, :] = np.exp(log_weights)

    missing_mass = 1.0 - weights.sum(axis=1)
    maximum_discrepancy = float(np.max(np.abs(missing_mass)))
    if maximum_discrepancy > 5.0e-12: raise RuntimeError(f"Weight normalization failed: {maximum_discrepancy:.3e}")
    return weights, maximum_discrepancy

# ============================================================
# 4. Exact inner Beta-prime expectations
# ============================================================
def inner_phi_values(n, k_max):
    k = np.arange(k_max + 1, dtype=float)
    a, b = k + 1.0, float(n)
    return 4.0 * a * b / ((a + b) * (a + b + 1.0))

def inner_psi_values(n, k_max):
    k = np.arange(k_max + 1, dtype=float)
    a, b = k + 1.0, float(n)
    mean_x = a / (a + b)
    values = 2.0 * betainc(a, b, 0.5) - 4.0 * mean_x * betainc(a + 1.0, b, 0.5) + 2.0 * mean_x - 1.0
    return np.clip(values, 0.0, 1.0)

# ============================================================
# 5. Bivariate operator
# ============================================================
def bivariate_approximations(n, z_grid):
    k_max, theoretical_tail = choose_k_max(n)
    weights, mass_error = outer_weights(n, z_grid, k_max)

    d_phi = weights @ inner_phi_values(n, k_max)
    d_psi = weights @ inner_psi_values(n, k_max)

    smooth_approximation = d_phi[:, None] * d_phi[None, :]
    nonsmooth_approximation = 0.5 * (d_psi[:, None] + d_psi[None, :])
    diagnostics = {"k_max": k_max, "tail": theoretical_tail, "mass_error": mass_error}

    return smooth_approximation, nonsmooth_approximation, diagnostics

# ============================================================
# 6. Metrics
# ============================================================
def calculate_metrics(target, approximation):
    error = target - approximation
    squared_error_sum = float(np.sum(error**2))
    target_variation = float(np.sum((target - np.mean(target)) ** 2))
    if target_variation <= 0.0: raise ValueError("R^2 cannot be calculated for a constant target.")

    return {
        "E_inf": float(np.max(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "R2": float(1.0 - squared_error_sum / target_variation),
    }

# ============================================================
# 7. Save results
# ============================================================
def save_csv(rows):
    output_path = OUTPUT_DIR / "bivariate_metrics.csv"
    fieldnames = ("function", "n", "E_inf", "RMSE", "MAE", "R2")
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "function": row["function"], "n": row["n"], "E_inf": f'{row["E_inf"]:.6f}',
                "RMSE": f'{row["RMSE"]:.6f}', "MAE": f'{row["MAE"]:.6f}', "R2": f'{row["R2"]:.6f}'
            })
    return output_path

# ============================================================
# 8. Main calculation
# ============================================================
def main():
    z_grid = np.linspace(Z_MIN, Z_MAX, GRID_SIZE)
    z1_grid, z2_grid = np.meshgrid(z_grid, z_grid, indexing="ij")
    targets = {"f_s": smooth_target(z1_grid, z2_grid), "f_ns": nonsmooth_target(z1_grid, z2_grid)}
    approximations, diagnostics = {"f_s": {}, "f_ns": {}}, {}

    for n in N_VALUES:
        smooth_app, nonsmooth_app, info = bivariate_approximations(n, z_grid)
        approximations["f_s"][n] = smooth_app
        approximations["f_ns"][n] = nonsmooth_app
        diagnostics[n] = info

    rows = []
    for function_name in ("f_s", "f_ns"):
        for n in N_VALUES:
            metric_values = calculate_metrics(targets[function_name], approximations[function_name][n])
            rows.append({"function": function_name, "n": n, **metric_values})

    print("\nBivariate approximation metrics\n" + "-" * 78)
    print(f'{"Function":<10}{"n":>6}{"E_inf":>14}{"RMSE":>14}{"MAE":>14}{"R2":>14}\n' + "-" * 78)
    for row in rows:
        print(f'{row["function"]:<10}{row["n"]:>6d}{row["E_inf"]:>14.6f}{row["RMSE"]:>14.6f}{row["MAE"]:>14.6f}{row["R2"]:>14.6f}')
    print("-" * 78)

    print("\nSeries-truncation diagnostics")
    for n in N_VALUES:
        info = diagnostics[n]
        print(f'n={n:>2d}: k_max={info["k_max"]:>3d}, tail={info["tail"]:.3e}, mass error={info["mass_error"]:.3e}')

    output_path = save_csv(rows)
    print(f"\nCSV saved to: {output_path}")

if __name__ == "__main__":
    main()