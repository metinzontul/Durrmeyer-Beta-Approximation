# -*- coding: utf-8 -*-
"""
Script: dbsl_benchmark_plot.py
Author: Metin Zontul
Description: Generates high-resolution (600 DPI) comparative benchmark plots 
             evaluating the Durrmeyer-Beta operator against ARIMA and LSTM baselines.
"""

import matplotlib.pyplot as plt

# ==========================================
# 1. DEFINITIVE BENCHMARK DATA
# ==========================================
noise_levels = [2.0, 4.0, 8.0, 12.0, 16.0]

# NRMSE Values (Lower is better)
nrmse_raw  = [0.0266, 0.0531, 0.1063, 0.1594, 0.2125]
nrmse_ar   = [0.0572, 0.0492, 0.0799, 0.0872, 0.1050]
nrmse_lstm = [0.0237, 0.0284, 0.0558, 0.1049, 0.1616]
nrmse_db   = [0.0143, 0.0188, 0.0306, 0.0437, 0.0571]

# NMAE Values (Lower is better)
nmae_raw  = [0.0213, 0.0426, 0.0853, 0.1279, 0.1705]
nmae_ar   = [0.0207, 0.0298, 0.0479, 0.0635, 0.0777]
nmae_lstm = [0.0177, 0.0220, 0.0432, 0.0796, 0.1234]
nmae_db   = [0.0117, 0.0148, 0.0230, 0.0320, 0.0415]

# R² Scores (Higher is better)
r2_raw  = [0.9890, 0.9561, 0.8242, 0.6045, 0.2969]
r2_ar   = [0.9491, 0.9624, 0.9005, 0.8815, 0.8282]
r2_lstm = [0.9912, 0.9874, 0.9516, 0.8286, 0.5932]
r2_db   = [0.9968, 0.9945, 0.9854, 0.9703, 0.9493]

# ==========================================
# 2. ACADEMIC PLOT STYLE SETTINGS
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 13,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

def create_benchmark_plot(x_data, raw_y, arima_y, lstm_y, dbsl_y, ylabel, title, filename, legend_loc='best', is_r2=False):
    """Standardized plotting function for benchmark metrics."""
    plt.figure(figsize=(8, 6), dpi=600)
    
    # Plot lines with distinct styles for accessibility
    plt.plot(x_data, raw_y, 's--', color='gray', linewidth=2.5, markersize=9, label='Raw Noisy Data')
    plt.plot(x_data, arima_y, '^-.', color='coral', linewidth=2.5, markersize=9, label='ARIMA Baseline')
    plt.plot(x_data, lstm_y, 'v:', color='mediumorchid', linewidth=2.5, markersize=9, label='LSTM Autoencoder')
    plt.plot(x_data, dbsl_y, 'o-', color='teal', linewidth=3, markersize=9, label='Durrmeyer-Beta (n=80)')
    
    # Axes and Titles
    plt.xlabel('Noise Level (Standard Deviation)')
    plt.ylabel(ylabel)
    plt.title(title, pad=15)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Lock X-axis to measurement points
    plt.xticks(x_data)
    
    # Fix Y-axis for R2 visual consistency
    if is_r2:
        plt.ylim([0.2, 1.05])
        
    plt.legend(loc=legend_loc, framealpha=0.9, edgecolor='lightgray')
    plt.tight_layout()
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=600)
    plt.close()
    print(f"--> Successfully saved: {filename}")

if __name__ == "__main__":
    create_benchmark_plot(noise_levels, nrmse_raw, nrmse_ar, nrmse_lstm, nrmse_db, 
                          'Normalized RMSE (NRMSE)', 'Error Degradation (RMSE) vs. Noise Level', 'Benchmark_RMSE_600dpi.png', 'upper left')
    create_benchmark_plot(noise_levels, nmae_raw, nmae_ar, nmae_lstm, nmae_db, 
                          'Normalized MAE (NMAE)', 'Error Degradation (MAE) vs. Noise Level', 'Benchmark_MAE_600dpi.png', 'upper left')
    create_benchmark_plot(noise_levels, r2_raw, r2_ar, r2_lstm, r2_db, 
                          'Coefficient of Determination (R²)', 'Structural Integrity (R²) vs. Noise Level', 'Benchmark_R2_600dpi.png', 'lower left', True)