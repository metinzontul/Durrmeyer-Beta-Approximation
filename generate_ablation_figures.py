# -*- coding: utf-8 -*-
"""
Script: generate_ablation_figures.py
Author: Metin Zontul
Date:   2026 October
Description: Reads the reported mean metrics from 'ablation logs.txt'
             and dynamically generates the three benchmark figures 
             (NRMSE, NMAE, and R2) for transparent reproducibility.

The log-file model names are preserved.
Only the displayed LSTM legend label is shortened for aesthetics.
"""

from pathlib import Path
import re
import matplotlib.pyplot as plt

# =============================================================================
# 1. FILE LOCATIONS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "ablation logs.txt"

# =============================================================================
# 2. PLOT SETTINGS
# =============================================================================
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 600,
    "savefig.dpi": 600,
})

# These names must exactly match the entries in the log file.
MODELS = [
    "Raw Noisy Data",
    "ARIMA Baseline",
    "LSTM Autoencoder",
    "Gaussian Kernel Smoother",
    "Durrmeyer-Beta (n=80)",
]

PLOT_CONFIG = {
    "Raw Noisy Data": {
        "marker": "s", "linestyle": "--", "color": "#7f7f7f", "label": "Raw noisy data"
    },
    "ARIMA Baseline": {
        "marker": "^", "linestyle": "-.", "color": "#ff7f0e", "label": "ARIMA baseline"
    },
    "LSTM Autoencoder": {
        "marker": "v", "linestyle": ":", "color": "#9467bd", "label": "LSTM reconstruction"
    },
    "Gaussian Kernel Smoother": {
        "marker": "D", "linestyle": "-", "color": "#2ca02c", "label": "Gaussian kernel"
    },
    "Durrmeyer-Beta (n=80)": {
        "marker": "o", "linestyle": "-", "color": "#1f77b4", "label": r"Durrmeyer--Beta ($D_{80}$)"
    },
}

# =============================================================================
# 3. READ MEAN METRICS FROM THE LOG
# =============================================================================
def read_log(log_file):
    if not log_file.is_file():
        raise FileNotFoundError(
            f"Log file not found:\n{log_file}\n"
            "Place 'ablation logs.txt' in the same folder as this script."
        )

    noise_levels = []
    data = {model: {"NRMSE": [], "NMAE": [], "R2": []} for model in MODELS}

    print(f"Reading data from: {log_file.name}")

    with log_file.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            if "[ RESULTS FOR NOISE LEVEL: Std =" in line:
                match = re.search(r"Std\s*=\s*([\d.]+)", line)
                if match:
                    noise_levels.append(float(match.group(1)))
                continue

            stripped = line.strip()
            for model in MODELS:
                if not stripped.startswith(model):
                    continue

                parts = stripped.split("|")
                if len(parts) != 4:
                    continue

                means = [float(part.split("±")[0].strip()) for part in parts[1:]]

                for metric, value in zip(("NRMSE", "NMAE", "R2"), means):
                    data[model][metric].append(value)
                break

    if not noise_levels:
        raise ValueError("No noise-level result blocks were found in the log.")

    expected_length = len(noise_levels)
    for model in MODELS:
        for metric in ("NRMSE", "NMAE", "R2"):
            actual_length = len(data[model][metric])
            if actual_length != expected_length:
                raise ValueError(
                    f"Incomplete data for '{model}', metric '{metric}': "
                    f"expected {expected_length}, found {actual_length}."
                )

    print(f"Successfully detected noise levels: {noise_levels}")
    return noise_levels, data

# =============================================================================
# 4. DRAW AND SAVE A FIGURE
# =============================================================================
def plot_metric(noise_levels, data, metric_name, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(9, 6))

    for model in MODELS:
        style = PLOT_CONFIG[model]
        ax.plot(
            noise_levels,
            data[model][metric_name],
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            linewidth=2.5,
            markersize=8,
            label=style["label"],
        )

    ax.set_xlabel(r"Noise standard deviation $\sigma$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=14)
    ax.set_xticks(noise_levels)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="best", frameon=True)
    
    fig.tight_layout()
    output_file = BASE_DIR / filename
    
    fig.savefig(output_file, dpi=600, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    print(f"Saved figure: {output_file.name}")

# =============================================================================
# 5. MAIN
# =============================================================================
def main():
    noise_levels, data = read_log(LOG_FILE)

    plot_metric(
        noise_levels=noise_levels,
        data=data,
        metric_name="NRMSE",
        ylabel="Mean normalized RMSE (NRMSE)",
        title="Mean NRMSE vs. Noise Level",
        filename="Figure_Synthetic_NRMSE.png",
    )

    plot_metric(
        noise_levels=noise_levels,
        data=data,
        metric_name="NMAE",
        ylabel="Mean normalized MAE (NMAE)",
        title="Mean NMAE vs. Noise Level",
        filename="Figure_Synthetic_NMAE.png",
    )

    plot_metric(
        noise_levels=noise_levels,
        data=data,
        metric_name="R2",
        ylabel=r"Mean coefficient of determination ($R^2$)",
        title=r"Mean $R^2$ vs. Noise Level",
        filename="Figure_Synthetic_R2.png",
    )

    print("\nProcess complete. All figures generated successfully.")

if __name__ == "__main__":
    main()