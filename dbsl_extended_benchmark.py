# -*- coding: utf-8 -*-
"""
Script: dbsl_extended_benchmark.py
Author: Metin Zontul
Date:   2026 October
Description: Executes a parallelized comparative robustness analysis of the 
             Durrmeyer-Beta operator against Gaussian Kernel, ARIMA, and LSTM Autoencoder.
             Includes per-signal statistical variance evaluation (NRMSE, NMAE, R2) over 
             30 independent realizations to ensure rigorous statistical reporting.
"""

import os
import warnings
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed

# ==========================================
# 1. DURRMEYER-BETA OPERATOR
# ==========================================
def durrmeyer_beta_reconstruct(noisy_signal, n, z_grid, tol=1e-14):
    # Local imports are used to prevent parallel processing serialization issues
    import math
    import numpy as np
    import warnings
    from scipy.integrate import IntegrationWarning, quad
    from scipy.special import betaln
    from scipy.interpolate import interp1d

    warnings.simplefilter("ignore", category=IntegrationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="scipy.integrate")

    t_points = np.linspace(0, 2, len(noisy_signal))
    f_interp = interp1d(t_points, noisy_signal, kind='cubic', bounds_error=False, fill_value=(noisy_signal[0], noisy_signal[-1]))
    
    global_k_max = 0
    for z in z_grid:
        k = 0
        p_val = 2.0 ** (-n * z)
        mass = p_val
        while 1.0 - mass >= tol and k < 2000:
            k += 1
            p_val = p_val * (n * z + k - 1) / (2.0 * k)
            mass += p_val
        if k > global_k_max:
            global_k_max = k
            
    A_nk = np.zeros(global_k_max + 1)
    for k in range(global_k_max + 1):
        beta_ln_val = betaln(k + 1, n)
        
        def integrand(x):
            if x <= 1e-10 or x >= 1.0 - 1e-10:
                return 0.0
            t = x / (1.0 - x)
            if t > 2.0: t = 2.0
            log_val = k * math.log(x) + (n - 1) * math.log(1.0 - x) - beta_ln_val
            return math.exp(log_val) * f_interp(t)
            
        mode_x = 1e-5 if k == 0 else k / (k + n - 1)
        integral_val, _ = quad(integrand, 0, 1, points=[mode_x], epsabs=1e-12, epsrel=1e-11, limit=100)
        A_nk[k] = integral_val

    reconstructed_signal = np.zeros_like(z_grid)
    for i, z in enumerate(z_grid):
        p_val = 2.0 ** (-n * z)
        val = p_val * A_nk[0]
        for k in range(1, global_k_max + 1):
            p_val = p_val * (n * z + k - 1) / (2.0 * k)
            val += p_val * A_nk[k]
        reconstructed_signal[i] = val
        
    return reconstructed_signal

# ==========================================
# 2. NON-PARAMETRIC BASELINE: GAUSSIAN KERNEL SMOOTHER
# ==========================================
def kernel_smoother_reconstruct(noisy_signal, z_grid, bandwidth=0.05):
    import numpy as np
    
    z_train = np.linspace(0, 2, len(noisy_signal))
    reconstructed = np.zeros_like(z_grid)
    
    for i, z in enumerate(z_grid):
        weights = np.exp(-0.5 * ((z_train - z) / bandwidth) ** 2)
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            reconstructed[i] = np.sum(weights * noisy_signal) / weight_sum
        else:
            reconstructed[i] = noisy_signal[i] if i < len(noisy_signal) else noisy_signal[-1]
            
    return reconstructed

# ==========================================
# 3. ARIMA BASELINE
# ==========================================
def arima_reconstruct(noisy_signal):
    import warnings
    from statsmodels.tsa.arima.model import ARIMA
    warnings.filterwarnings("ignore")
    try:
        model = ARIMA(noisy_signal, order=(2, 0, 2))
        res = model.fit()
        return res.fittedvalues
    except:
        return noisy_signal

# ==========================================
# 4. LSTM AUTOENCODER BASELINE
# ==========================================
def lstm_reconstruct(noisy_signal):
    import warnings
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.preprocessing import MinMaxScaler
    warnings.filterwarnings("ignore")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    signal_scaled = scaler.fit_transform(noisy_signal.reshape(-1, 1))
    
    torch.manual_seed(42)
    x_tensor = torch.tensor(signal_scaled, dtype=torch.float32).view(1, -1, 1)
    
    class LSTMAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
            self.linear = nn.Linear(16, 1)
            
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.linear(out)
            
    model = LSTMAutoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for _ in range(100):
        optimizer.zero_grad()
        pred = model(x_tensor)
        loss = criterion(pred, x_tensor)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        pred_scaled = model(x_tensor).view(-1, 1).numpy()
        
    pred_original = scaler.inverse_transform(pred_scaled).flatten()
    return pred_original

# ==========================================
# 5. DATA GENERATION AND PARALLEL EXECUTORS
# ==========================================
def generate_spectroscopy_data(noise_std, num_signals=30, length=100):
    np.random.seed(42)
    z = np.linspace(0, 2, length)
    clean_signals, noisy_signals = [], []
    for _ in range(num_signals):
        moisture_index = np.random.uniform(10.0, 50.0)
        base_signal = moisture_index * ((np.pi * z) / (1 + z**2))
        noise = np.random.normal(0, noise_std, length)
        clean_signals.append(base_signal)
        noisy_signals.append(base_signal + noise)
    return np.array(clean_signals), np.array(noisy_signals), z

def apply_operator_parallel(noisy_signals, z_grid, n_param=80):
    print("  -> Running Durrmeyer-Beta Operator...")
    return np.array(Parallel(n_jobs=-1, verbose=0)(delayed(durrmeyer_beta_reconstruct)(sig, n=n_param, z_grid=z_grid) for sig in noisy_signals))

def apply_kernel_parallel(noisy_signals, z_grid, bandwidth=0.05):
    print("  -> Running Gaussian Kernel Smoother...")
    return np.array(Parallel(n_jobs=-1, verbose=0)(delayed(kernel_smoother_reconstruct)(sig, z_grid, bandwidth) for sig in noisy_signals))

def apply_arima_parallel(noisy_signals):
    print("  -> Running ARIMA Baseline...")
    return np.array(Parallel(n_jobs=-1, verbose=0)(delayed(arima_reconstruct)(sig) for sig in noisy_signals))

def apply_lstm_parallel(noisy_signals):
    print("  -> Running LSTM Autoencoder Baseline...")
    return np.array(Parallel(n_jobs=-1, verbose=0)(delayed(lstm_reconstruct)(sig) for sig in noisy_signals))

# ==========================================
# 6. STATISTICAL ROBUSTNESS ANALYSIS
# ==========================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    print("\n" + "="*140)
    print(" DBSL vs NON-PARAMETRIC KERNEL vs ARIMA vs LSTM: STATISTICAL NOISE ROBUSTNESS ANALYSIS")
    print("="*140)
    
    noise_levels = [2.0, 4.0, 8.0, 12.0, 16.0]
    n_parameter = 80
    
    for noise_std in noise_levels:
        print(f"\n[ INITIALIZING NOISE LEVEL: Std = {noise_std:.1f} ]")
        clean_X, noisy_X, z_grid = generate_spectroscopy_data(noise_std=noise_std, num_signals=30, length=100)
        
        pred_X_DBSL = apply_operator_parallel(noisy_X, z_grid, n_param=n_parameter)
        pred_X_KERN = apply_kernel_parallel(noisy_X, z_grid, bandwidth=0.05)
        pred_X_ARIMA = apply_arima_parallel(noisy_X)
        pred_X_LSTM = apply_lstm_parallel(noisy_X)
        
        # Per-signal normalization to ensure scale-invariant metric comparisons
        scaler = MinMaxScaler()
        clean_norm = scaler.fit_transform(clean_X.T).T
        noisy_norm = scaler.transform(noisy_X.T).T
        
        pred_DBSL_norm = scaler.transform(pred_X_DBSL.T).T
        pred_KERN_norm = scaler.transform(pred_X_KERN.T).T
        pred_ARIMA_norm = scaler.transform(pred_X_ARIMA.T).T
        pred_LSTM_norm = scaler.transform(pred_X_LSTM.T).T
        
        # Dictionary to accumulate per-signal metrics (NRMSE, NMAE, and R2)
        metrics = {'raw': {'nrmse': [], 'nmae': [], 'r2': []},
                   'ar': {'nrmse': [], 'nmae': [], 'r2': []},
                   'ls': {'nrmse': [], 'nmae': [], 'r2': []},
                   'kn': {'nrmse': [], 'nmae': [], 'r2': []},
                   'db': {'nrmse': [], 'nmae': [], 'r2': []}}
                   
        for i in range(len(clean_norm)):
            c, n = clean_norm[i], noisy_norm[i]
            p_ar, p_ls = pred_ARIMA_norm[i], pred_LSTM_norm[i]
            p_kn, p_db = pred_KERN_norm[i], pred_DBSL_norm[i]
            
            metrics['raw']['nrmse'].append(np.sqrt(mean_squared_error(c, n)))
            metrics['raw']['nmae'].append(mean_absolute_error(c, n))
            metrics['raw']['r2'].append(r2_score(c, n))
            
            metrics['ar']['nrmse'].append(np.sqrt(mean_squared_error(c, p_ar)))
            metrics['ar']['nmae'].append(mean_absolute_error(c, p_ar))
            metrics['ar']['r2'].append(r2_score(c, p_ar))
            
            metrics['ls']['nrmse'].append(np.sqrt(mean_squared_error(c, p_ls)))
            metrics['ls']['nmae'].append(mean_absolute_error(c, p_ls))
            metrics['ls']['r2'].append(r2_score(c, p_ls))
            
            metrics['kn']['nrmse'].append(np.sqrt(mean_squared_error(c, p_kn)))
            metrics['kn']['nmae'].append(mean_absolute_error(c, p_kn))
            metrics['kn']['r2'].append(r2_score(c, p_kn))
            
            metrics['db']['nrmse'].append(np.sqrt(mean_squared_error(c, p_db)))
            metrics['db']['nmae'].append(mean_absolute_error(c, p_db))
            metrics['db']['r2'].append(r2_score(c, p_db))
            
        # Compile and print results formatted as Mean +/- Standard Deviation
        print(f"\n[ RESULTS FOR NOISE LEVEL: Std = {noise_std:.1f} ]")
        print("-" * 140)
        print(f"{'Model / Approach':<28} | {'NRMSE (↓) [Mean ± Std]':<32} | {'NMAE (↓) [Mean ± Std]':<32} | {'R² Score (↑) [Mean ± Std]':<32}")
        print("-" * 140)
        keys = ['raw', 'ar', 'ls', 'kn', 'db']
        names = ['Raw Noisy Data', 'ARIMA Baseline', 'LSTM Autoencoder', 'Gaussian Kernel Smoother', 'Durrmeyer-Beta (n=80)']
        for key, name in zip(keys, names):
            mean_nrmse, std_nrmse = np.mean(metrics[key]['nrmse']), np.std(metrics[key]['nrmse'])
            mean_nmae, std_nmae = np.mean(metrics[key]['nmae']), np.std(metrics[key]['nmae'])
            mean_r2, std_r2 = np.mean(metrics[key]['r2']), np.std(metrics[key]['r2'])
            print(f"{name:<28} | {mean_nrmse:.4f} ± {std_nrmse:.4f} {'':<13} | {mean_nmae:.4f} ± {std_nmae:.4f} {'':<13} | {mean_r2:.4f} ± {std_r2:.4f}")
        print("="*140)