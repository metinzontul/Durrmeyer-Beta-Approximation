# -*- coding: utf-8 -*-
"""
Script: generate_bivariate_approximation_figures.py
Author: Seda Karateke
Description: Generates professional bivariate approximation figures for the 
             tensor-product Durrmeyer-Beta operator.

Test functions:
    f_s(z1,z2)  = phi(z1) phi(z2)
    f_ns(z1,z2) = [psi(z1) + psi(z2)] / 2

Files created:
    figure5_bivariate_smooth_approximations.png
    figure6_bivariate_nonsmooth_approximations.png
    figure7_bivariate_absolute_errors.png
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
from scipy.special import betainc, gammaln
from scipy.stats import nbinom

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent
N_VALUES = (5, 10, 20, 40, 80)
Z_MIN, Z_MAX = 0.0, 2.0
PLOT_GRID_SIZE = 101
TAIL_TOL = 1.0e-14
OUTPUT_DPI = 600
ERROR_COLOR_GAMMA = 0.55
ERROR_CONTOUR_LEVELS = (0.025, 0.050, 0.100, 0.200, 0.300)

ACCENT_COLORS = {"target": "#2F2F2F", 5: "#D55E00", 10: "#E69F00", 20: "#009E73", 40: "#0072B2", 80: "#7B2CBF"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 10.5,
    "axes.labelsize": 9.0, "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
    "figure.titlesize": 15.0, "axes.linewidth": 0.7,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})

# -----------------------------------------------------------------------------
# Test functions
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Durrmeyer--Beta operator
# -----------------------------------------------------------------------------
def choose_k_max(n):
    r_max = n * Z_MAX
    quantile = nbinom.ppf(1.0 - TAIL_TOL, r_max, 0.5)
    if not np.isfinite(quantile): raise RuntimeError(f"Finite truncation index not found for n={n}.")
    k_max = int(np.ceil(quantile))
    tail = float(nbinom.sf(k_max, r_max, 0.5))
    if tail > 5.0 * TAIL_TOL: raise RuntimeError(f"Tail {tail:.3e} exceeds tolerance for n={n}.")
    return k_max

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
    max_mass_error = float(np.max(np.abs(1.0 - weights.sum(axis=1))))
    if max_mass_error > 5.0e-12: raise RuntimeError(f"Weight normalization failed: discrepancy={max_mass_error:.3e}.")
    return weights

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

def calculate_all_surfaces(z_grid):
    z1_grid, z2_grid = np.meshgrid(z_grid, z_grid, indexing="ij")
    targets = {"smooth": smooth_target(z1_grid, z2_grid), "nonsmooth": nonsmooth_target(z1_grid, z2_grid)}
    approximations = {"smooth": {}, "nonsmooth": {}}
    errors = {"smooth": {}, "nonsmooth": {}}

    for n in N_VALUES:
        k_max = choose_k_max(n)
        weights = outer_weights(n, z_grid, k_max)
        d_phi = weights @ inner_phi_values(n, k_max)
        d_psi = weights @ inner_psi_values(n, k_max)

        smooth_app = d_phi[:, None] * d_phi[None, :]
        nonsmooth_app = 0.5 * (d_psi[:, None] + d_psi[None, :])

        approximations["smooth"][n] = smooth_app
        approximations["nonsmooth"][n] = nonsmooth_app
        errors["smooth"][n] = np.abs(smooth_app - targets["smooth"])
        errors["nonsmooth"][n] = np.abs(nonsmooth_app - targets["nonsmooth"])

    return z1_grid, z2_grid, targets, approximations, errors

# -----------------------------------------------------------------------------
# Figure styling helpers
# -----------------------------------------------------------------------------
def style_3d_axis(ax):
    ax.set_xlim(Z_MIN, Z_MAX); ax.set_ylim(Z_MIN, Z_MAX); ax.set_zlim(0.0, 1.02)
    ax.set_xticks((0.0, 1.0, 2.0)); ax.set_yticks((0.0, 1.0, 2.0)); ax.set_zticks((0.0, 0.5, 1.0))
    ax.set_xlabel(r"$z_1$", labelpad=1); ax.set_ylabel(r"$z_2$", labelpad=1); ax.set_zlabel("value", labelpad=2)
    ax.tick_params(axis="both", which="major", pad=0, length=2.0)
    ax.view_init(elev=28, azim=-136); ax.set_box_aspect((1.0, 1.0, 0.72))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.97, 0.98, 1.00, 1.0)); axis.pane.set_edgecolor((0.78, 0.80, 0.84, 1.0))
        axis._axinfo["grid"].update({"color": (0.72, 0.74, 0.78, 0.42), "linewidth": 0.45})

def make_surface_figure(z1_grid, z2_grid, target, approximations, function_key, function_formula, output_filename):
    fig = plt.figure(figsize=(13.2, 8.5))
    axes = [fig.add_subplot(2, 3, position, projection="3d") for position in range(1, 7)]
    panels = [("target", target)] + [(n, approximations[n]) for n in N_VALUES]
    panel_letters = ("a", "b", "c", "d", "e", "f")
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("viridis")

    for ax, letter, (panel_key, values) in zip(axes, panel_letters, panels):
        ax.plot_surface(z1_grid, z2_grid, values, cmap=cmap, norm=norm, rstride=2, cstride=2, linewidth=0.08, edgecolor=(0.10, 0.12, 0.18, 0.16), antialiased=True, shade=True)
        title = rf"({letter}) Target ${function_key}$" if panel_key == "target" else rf"({letter}) $D_{{{panel_key},{panel_key}}}({function_key})$"
        accent = ACCENT_COLORS["target"] if panel_key == "target" else ACCENT_COLORS[panel_key]
        ax.set_title(title, pad=5.5, color=accent, fontweight="semibold")
        style_3d_axis(ax)

    fig.subplots_adjust(left=0.025, right=0.895, bottom=0.045, top=0.875, wspace=0.045, hspace=0.12)
    fig.suptitle(f"Bivariate Durrmeyer--Beta approximation: {function_formula}", x=0.46, y=0.975, color="#172033", fontweight="bold")
    colorbar_axis = fig.add_axes((0.92, 0.17, 0.018, 0.66))
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_axis)
    colorbar.set_label("Function value", labelpad=8)
    colorbar.set_ticks((0.0, 0.25, 0.50, 0.75, 1.0)); colorbar.outline.set_linewidth(0.65)
    
    output_path = OUTPUT_DIR / output_filename
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output_path

def make_error_figure(z_grid, errors):
    fig, axes = plt.subplots(2, 5, figsize=(15.6, 6.2), sharex=True, sharey=True, constrained_layout=False)
    max_error = max(float(np.max(values)) for family in errors.values() for values in family.values())
    norm = PowerNorm(gamma=ERROR_COLOR_GAMMA, vmin=0.0, vmax=max_error, clip=True)
    cmap = LinearSegmentedColormap.from_list("vivid_absolute_error", ("#F7FCFF", "#BDEBFF", "#35C9E8", "#20B779", "#F4E61E", "#FF8A1F", "#D7191C"), N=256)
    row_info = (("smooth", r"$f_{\mathrm{s}}$"), ("nonsmooth", r"$f_{\mathrm{ns}}$"))

    for row_index, (family, row_label) in enumerate(row_info):
        for column_index, n in enumerate(N_VALUES):
            ax = axes[row_index, column_index]; values = errors[family][n]
            image = ax.imshow(values.T, origin="lower", extent=(Z_MIN, Z_MAX, Z_MIN, Z_MAX), cmap=cmap, norm=norm, interpolation="bilinear", aspect="equal")
            e_inf = float(np.max(values))
            visible_levels = tuple(level for level in ERROR_CONTOUR_LEVELS if level < e_inf)
            if visible_levels: ax.contour(z_grid, z_grid, values.T, levels=visible_levels, colors="#16324F", linewidths=0.55, alpha=0.62)
            if row_index == 0: ax.set_title(rf"$n_1=n_2={n}$", pad=7, color=ACCENT_COLORS[n], fontweight="bold")
            if column_index == 0: ax.set_ylabel(r"$z_2$", labelpad=4)
            if row_index == 1: ax.set_xlabel(r"$z_1$", labelpad=3)
            ax.set_xticks((0.0, 1.0, 2.0)); ax.set_yticks((0.0, 1.0, 2.0)); ax.grid(False)
            ax.text(0.06, 0.92, rf"$E_\infty={e_inf:.4f}$", transform=ax.transAxes, ha="left", va="top", fontsize=8.2, color="#10243E", bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": ACCENT_COLORS[n], "linewidth": 0.75, "alpha": 0.94})
            for spine in ax.spines.values(): spine.set_color(ACCENT_COLORS[n]); spine.set_linewidth(1.15)

    fig.subplots_adjust(left=0.075, right=0.90, bottom=0.105, top=0.845, wspace=0.10, hspace=0.10)
    fig.text(0.018, 0.65, r"Smooth $f_{\mathrm{s}}$", rotation=90, ha="center", va="center", color="#172033", fontweight="semibold")
    fig.text(0.018, 0.275, r"Nonsmooth $f_{\mathrm{ns}}$", rotation=90, ha="center", va="center", color="#172033", fontweight="semibold")
    fig.suptitle("Absolute-error distributions on $[0,2]^2$", x=0.47, y=0.965, color="#172033", fontweight="bold")
    fig.text(0.47, 0.905, rf"Common absolute-error range in every panel; power-scaled colors ($\gamma={ERROR_COLOR_GAMMA:.2f}$)", ha="center", va="center", fontsize=9.2, color="#46546A")
    
    colorbar_axis = fig.add_axes((0.925, 0.18, 0.014, 0.64))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar_ticks = tuple(value for value in (0.0, 0.025, 0.050, 0.100, 0.200, 0.300) if value < max_error) + (max_error,)
    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels([f"{value:.3f}" for value in colorbar_ticks[:-1]] + [f"{max_error:.3f}"])
    colorbar.set_label(rf"Absolute error (power scale, $\gamma={ERROR_COLOR_GAMMA:.2f}$)", labelpad=8)
    colorbar.outline.set_linewidth(0.65)
    
    output_path = OUTPUT_DIR / "figure7_bivariate_absolute_errors.png"
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output_path

def main():
    z_grid = np.linspace(Z_MIN, Z_MAX, PLOT_GRID_SIZE)
    z1_grid, z2_grid, targets, approximations, errors = calculate_all_surfaces(z_grid)
    
    output_paths = [
        make_surface_figure(z1_grid, z2_grid, targets["smooth"], approximations["smooth"], r"f_{\mathrm{s}}", r"$f_{\mathrm{s}}(z_1,z_2)=\phi(z_1)\phi(z_2)$", "figure5_bivariate_smooth_approximations.png"),
        make_surface_figure(z1_grid, z2_grid, targets["nonsmooth"], approximations["nonsmooth"], r"f_{\mathrm{ns}}", r"$f_{\mathrm{ns}}(z_1,z_2)=[\psi(z_1)+\psi(z_2)]/2$", "figure6_bivariate_nonsmooth_approximations.png"),
        make_error_figure(z_grid, errors)
    ]
    print("Created figures:")
    for path in output_paths: print(f"  {path}")

if __name__ == "__main__":
    main()