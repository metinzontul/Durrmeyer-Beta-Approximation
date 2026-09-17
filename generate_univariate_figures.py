# -*- coding: utf-8 -*-
"""
Script: generate_univariate_figures.py
Author: Seda Karateke
Description: Generates univariate numerical figures and accuracy tables for the 
             Durrmeyer-Beta operator approximation over [0,2].

Files created:
    figure1_univariate_approximation.png
    figure2_univariate_error.png
    figure3_second_test_approximation.png
    figure4_second_test_error.png
    univariate_metrics.csv
    univariate_metrics_table.tex
"""

from pathlib import Path
import csv
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.special import betaln, gammaln
from scipy.stats import nbinom

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

N_VALUES = (5, 10, 20, 40, 80)
Z_MIN, Z_MAX = 0.0, 2.0
GRID_SIZE = 1001
TAIL_TOL = 1.0e-14
QUAD_EPSABS, QUAD_EPSREL = 1.0e-12, 1.0e-11
QUAD_LIMIT = 800
OUTPUT_DPI = 600

COLORS = {"target": "#20242B", 5: "#D55E00", 10: "#E69F00", 20: "#009E73", 40: "#0072B2", 80: "#7B2CBF"}
LINE_STYLES = {5: "-", 10: "--", 20: "-.", 40: (0, (5, 1.5)), 80: (0, (1.2, 1.2))}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.0, "axes.titlesize": 12.0,
    "axes.labelsize": 11.0, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 9.0, "figure.titlesize": 15.0, "axes.linewidth": 0.8,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})

# -----------------------------------------------------------------------------
# Test functions
# -----------------------------------------------------------------------------
def first_test_function(z):
    z = np.asarray(z, dtype=float)
    return np.pi * z / (1.0 + z * z)

def second_test_function(z):
    z = np.asarray(z, dtype=float)
    return np.cos(np.pi * z) / (1.0 + z * z)

# -----------------------------------------------------------------------------
# Durrmeyer--Beta operator
# -----------------------------------------------------------------------------
def choose_k_max(n):
    largest_shape = n * Z_MAX
    quantile = nbinom.ppf(1.0 - TAIL_TOL, largest_shape, 0.5)
    if not np.isfinite(quantile): raise RuntimeError(f"No finite truncation index found for n={n}.")
    k_max = int(np.ceil(quantile))
    omitted_mass = float(nbinom.sf(k_max, largest_shape, 0.5))
    if omitted_mass > TAIL_TOL: raise RuntimeError(f"Tail {omitted_mass:.3e} exceeds tolerance for n={n}.")
    return k_max

def beta_expectation(n, k, function):
    log_beta = betaln(k + 1.0, float(n))
    def integrand(x):
        if x <= 0.0 or x >= 1.0: return 0.0
        t = x / (1.0 - x)
        log_density = k * np.log(x) + (n - 1.0) * np.log1p(-x) - log_beta
        return float(np.exp(log_density) * function(t))

    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        value, estimated_error = quad(integrand, 0.0, 1.0, epsabs=QUAD_EPSABS, epsrel=QUAD_EPSREL, limit=QUAD_LIMIT)

    if not np.isfinite(value): raise RuntimeError(f"Integral not finite for n={n}, k={k}.")
    if estimated_error > 5.0e-9: raise RuntimeError(f"Quadrature error failed accuracy: n={n}, k={k}.")
    return value

def inner_expectations(n, k_max, function):
    return np.array([beta_expectation(n, k, function) for k in range(k_max + 1)], dtype=float)

def outer_weights(n, z_grid, k_max):
    k = np.arange(k_max + 1, dtype=float)
    weight_matrix = np.zeros((z_grid.size, k_max + 1), dtype=float)

    for row, z in enumerate(z_grid):
        shape = n * float(z)
        if shape == 0.0:
            weight_matrix[row, 0] = 1.0
            continue
        log_weights = -shape * np.log(2.0) + gammaln(shape + k) - gammaln(shape) - gammaln(k + 1.0) - k * np.log(2.0)
        weight_matrix[row, :] = np.exp(log_weights)

    if float(np.max(np.abs(1.0 - weight_matrix.sum(axis=1)))) > 5.0e-12: raise RuntimeError("Weights failed normalization.")
    return weight_matrix

def evaluate_family(z_grid, function):
    target = function(z_grid)
    approximations, errors = {}, {}
    for n in N_VALUES:
        k_max = choose_k_max(n)
        expectations = inner_expectations(n, k_max, function)
        weights = outer_weights(n, z_grid, k_max)
        approximation = weights @ expectations
        approximations[n] = approximation
        errors[n] = np.abs(approximation - target)
    return target, approximations, errors

# -----------------------------------------------------------------------------
# Accuracy metrics
# -----------------------------------------------------------------------------
def calculate_metrics(target, approximation):
    residual = target - approximation
    absolute_residual = np.abs(residual)
    return {
        "E_inf": float(np.max(absolute_residual)),
        "RMSE": float(np.sqrt(np.mean(residual * residual))),
        "MAE": float(np.mean(absolute_residual)),
        "R2": 1.0 - float(np.sum(residual * residual)) / float(np.sum((target - np.mean(target)) ** 2)),
    }

def collect_metrics(targets, approximations):
    rows = []
    for family, display_name in (("f", "f"), ("g", "g")):
        for n in N_VALUES:
            metric_values = calculate_metrics(targets[family], approximations[family][n])
            rows.append({"function": display_name, "n": n, **metric_values})
    return rows

def write_metrics_csv(rows):
    output_path = OUTPUT_DIR / "univariate_metrics.csv"
    fieldnames = ("function", "n", "E_inf", "RMSE", "MAE", "R2")
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"function": row["function"], "n": row["n"], "E_inf": f"{row['E_inf']:.9f}", "RMSE": f"{row['RMSE']:.9f}", "MAE": f"{row['MAE']:.9f}", "R2": f"{row['R2']:.9f}"})
    return output_path

def write_metrics_latex(rows):
    lines = [
        r"\begin{table}[H]", r"\caption{Grid-based accuracy metrics for the univariate approximations on $K_1=[0,2]$.\label{tab:univariate-metrics}}",
        r"\centering", r"\small", r"\begin{tabular}{cccccc}", r"\toprule", r"Test function & $n$ & $E_{\infty}$ & RMSE & MAE & $R^2$\\", r"\midrule",
    ]
    for row_index, row in enumerate(rows):
        if row_index == len(N_VALUES): lines.append(r"\midrule")
        function_label = (r"$f$" if row["function"] == "f" else r"$g$") if row_index in (0, len(N_VALUES)) else ""
        lines.append(f"{function_label} & {row['n']} & {row['E_inf']:.6f} & {row['RMSE']:.6f} & {row['MAE']:.6f} & {row['R2']:.6f}\\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    output_path = OUTPUT_DIR / "univariate_metrics_table.tex"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path

# -----------------------------------------------------------------------------
# Figure styling
# -----------------------------------------------------------------------------
def style_axis(ax, z_grid):
    ax.set_xlim(Z_MIN, Z_MAX); ax.set_xticks(np.linspace(Z_MIN, Z_MAX, 5)); ax.set_xlabel(r"$z$")
    ax.set_facecolor("#FBFCFE"); ax.grid(True, which="major", color="#9AA6B2", linewidth=0.55, alpha=0.32); ax.set_axisbelow(True)
    for spine in ax.spines.values(): spine.set_color("#667085"); spine.set_linewidth(0.8)

def add_figure_legend(fig, handles, labels, columns=3):
    legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.895), ncol=columns, frameon=True, fancybox=True, columnspacing=1.5, handlelength=3.0, borderpad=0.55)
    legend.get_frame().set_facecolor("white"); legend.get_frame().set_edgecolor("#CDD3DB"); legend.get_frame().set_alpha(0.96)

def save_verified_figure(fig, output_filename):
    output_path = FIGURE_DIR / output_filename
    temporary_path = FIGURE_DIR / f".{Path(output_filename).stem}.tmp.png"
    fig.savefig(temporary_path, format="png", dpi=OUTPUT_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    if mpimg.imread(temporary_path).size == 0: raise RuntimeError(f"Generated PNG is empty: {temporary_path}")
    temporary_path.replace(output_path)
    return output_path

def make_approximation_figure(z_grid, target, approximations, formula, target_label, output_filename):
    fig, ax = plt.subplots(figsize=(10.2, 5.8)); fig.subplots_adjust(left=0.09, right=0.975, bottom=0.13, top=0.76)
    target_line, = ax.plot(z_grid, target, color=COLORS["target"], linewidth=2.8, label=target_label, zorder=10)
    handles, labels = [target_line], [target_label]
    for n in N_VALUES:
        line, = ax.plot(z_grid, approximations[n], color=COLORS[n], linestyle=LINE_STYLES[n], linewidth=1.9, label=rf"$D_{{{n}}}$")
        handles.append(line); labels.append(rf"$D_{{{n}}}$")
    
    all_values = np.concatenate([target] + [approximations[n] for n in N_VALUES])
    value_range = float(np.max(all_values) - np.min(all_values))
    padding = 0.07 * value_range if value_range > 0.0 else 0.1
    ax.set_ylim(float(np.min(all_values) - padding), float(np.max(all_values) + padding))
    ax.set_ylabel("Function value"); style_axis(ax, z_grid)
    fig.suptitle(f"Univariate Durrmeyer--Beta approximation: {formula}", x=0.5, y=0.975, color="#172033", fontweight="bold")
    add_figure_legend(fig, handles, labels, columns=3)
    return save_verified_figure(fig, output_filename)

def make_error_figure(z_grid, errors, formula, output_filename):
    fig, ax = plt.subplots(figsize=(10.2, 5.8)); fig.subplots_adjust(left=0.09, right=0.975, bottom=0.13, top=0.72)
    handles, labels, maximum_error = [], [], max(float(np.max(errors[n])) for n in N_VALUES)
    for n in N_VALUES:
        e_inf = float(np.max(errors[n]))
        line, = ax.plot(z_grid, errors[n], color=COLORS[n], linestyle=LINE_STYLES[n], linewidth=2.0, label=rf"$n={n}$, $E_\infty={e_inf:.4f}$")
        handles.append(line); labels.append(rf"$n={n}$, $E_\infty={e_inf:.4f}$")
    ax.set_ylim(0.0, 1.08 * maximum_error); ax.set_ylabel("Absolute error"); style_axis(ax, z_grid)
    fig.suptitle(f"Absolute errors: {formula}", x=0.5, y=0.975, color="#172033", fontweight="bold")
    add_figure_legend(fig, handles, labels, columns=3)
    return save_verified_figure(fig, output_filename)

def main():
    z_grid = np.linspace(Z_MIN, Z_MAX, GRID_SIZE)
    f_target, f_approximations, f_errors = evaluate_family(z_grid, first_test_function)
    g_target, g_approximations, g_errors = evaluate_family(z_grid, second_test_function)
    
    metric_rows = collect_metrics({"f": f_target, "g": g_target}, {"f": f_approximations, "g": g_approximations})
    output_paths = [
        make_approximation_figure(z_grid, f_target, f_approximations, r"$f(z)=\pi z/(1+z^2)$", r"Target $f$", "figure1_univariate_approximation.png"),
        make_error_figure(z_grid, f_errors, r"$f(z)=\pi z/(1+z^2)$", "figure2_univariate_error.png"),
        make_approximation_figure(z_grid, g_target, g_approximations, r"$g(z)=\cos(\pi z)/(1+z^2)$", r"Target $g$", "figure3_second_test_approximation.png"),
        make_error_figure(z_grid, g_errors, r"$g(z)=\cos(\pi z)/(1+z^2)$", "figure4_second_test_error.png"),
        write_metrics_csv(metric_rows), write_metrics_latex(metric_rows),
    ]
    print("Created files:"); [print(f"  {p}") for p in output_paths]

if __name__ == "__main__":
    main()