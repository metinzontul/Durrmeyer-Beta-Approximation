# -*- coding: utf-8 -*-
"""
Script: dbsl_vs_arima_lstm_benchmark.py
Author: Metin Zontul
Description: Executes a parallelized comparative robustness analysis of the 
             Durrmeyer-Beta operator against ARIMA and an LSTM Autoencoder under extreme noise.
"""

import os
import warnings
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from scipy.integrate import IntegrationWarning, quad
from scipy.special import betaln
from scipy.interpolate import interp1d
from statsmodels.tsa.arima.model import ARIMA

# ==========================================
# 1. DURRMEYER-BETA OPERATOR (Paper Compliant)
# ==========================================
def durrmeyer_beta_reconstruct(noisy_signal, n, z_grid, tol=1e-14):
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
# 2. ARIMA BASELINE
# ==========================================
def arima_reconstruct(noisy_signal):
    warnings.filterwarnings("ignore")
    try:
        model = ARIMA(noisy_signal, order=(2, 0, 2))
        res = model.fit()
        return res.fittedvalues
    except:
        return noisy_signal

# ==========================================
# 3. LSTM AUTOENCODER BASELINE (Normalization Fixed)
# ==========================================
def lstm_reconstruct(noisy_signal):
    warnings.filterwarnings("ignore")
    
    # CRITICAL STEP: SIGNAL NORMALIZATION (Prevents gradient explosion)
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
        
    # INVERSE NORMALIZATION (Return to original amplitude)
    pred_original = scaler.inverse_transform(pred_scaled).flatten()
    
    return pred_original

# ==========================================
# 4. DATA GENERATION AND PARALLEL EXECUTORS
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
    return np.array(Parallel(n_jobs=-1, verbose=0)(delayed(durrmeyer_beta_reconstruct)(sig, n=n_param, z_grid=z_grid) for sig in noisy_signals))

def apply_arima_parallel(noisy_signals):
    return np.array(Parallel(n_jobs=-1, verbose=0)(delayed(arima_reconstruct)(sig) for sig in noisy_signals))

def apply_lstm_parallel(noisy_signals):
    return np.array(Parallel(n_jobs=-1, verbose=0)(delayed(lstm_reconstruct)(sig) for sig in noisy_signals))

# ==========================================
# 5. ROBUSTNESS AND BASELINE ANALYSIS
# ==========================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    print("\n" + "="*85)
    print(" DBSL vs ARIMA vs LSTM: NOISE ROBUSTNESS ANALYSIS")
    print(" Metrics: NRMSE, NMAE, and R²")
    print("="*85)
    
    noise_levels = [2.0, 4.0, 8.0, 12.0, 16.0]
    n_parameter = 80
    
    for noise_std in noise_levels:
        clean_X, noisy_X, z_grid = generate_spectroscopy_data(noise_std=noise_std, num_signals=30, length=100)
        
        # Execute models
        pred_X_DBSL = apply_operator_parallel(noisy_X, z_grid, n_param=n_parameter)
        pred_X_ARIMA = apply_arima_parallel(noisy_X)
        pred_X_LSTM = apply_lstm_parallel(noisy_X)
        
        # Normalize predictions to common [0,1] scale for fair evaluation
        scaler = MinMaxScaler()
        clean_norm = scaler.fit_transform(clean_X.flatten().reshape(-1, 1)).flatten()
        noisy_norm = scaler.transform(noisy_X.flatten().reshape(-1, 1)).flatten()
        
        pred_DBSL_norm = scaler.transform(pred_X_DBSL.flatten().reshape(-1, 1)).flatten()
        pred_ARIMA_norm = scaler.transform(pred_X_ARIMA.flatten().reshape(-1, 1)).flatten()
        pred_LSTM_norm = scaler.transform(pred_X_LSTM.flatten().reshape(-1, 1)).flatten()
        
        # Calculate Metrics
        nrmse_raw = np.sqrt(mean_squared_error(clean_norm, noisy_norm))
        nmae_raw = mean_absolute_error(clean_norm, noisy_norm)
        r2_raw = r2_score(clean_norm, noisy_norm)

        nrmse_ar = np.sqrt(mean_squared_error(clean_norm, pred_ARIMA_norm))
        nmae_ar = mean_absolute_error(clean_norm, pred_ARIMA_norm)
        r2_ar = r2_score(clean_norm, pred_ARIMA_norm)

        nrmse_ls = np.sqrt(mean_squared_error(clean_norm, pred_LSTM_norm))
        nmae_ls = mean_absolute_error(clean_norm, pred_LSTM_norm)
        r2_ls = r2_score(clean_norm, pred_LSTM_norm)

        nrmse_db = np.sqrt(mean_squared_error(clean_norm, pred_DBSL_norm))
        nmae_db = mean_absolute_error(clean_norm, pred_DBSL_norm)
        r2_db = r2_score(clean_norm, pred_DBSL_norm)
        
        # Evaluation Report
        print(f"\n[ NOISE LEVEL: Std = {noise_std:.1f} ]")
        print("-" * 85)
        print(f"{'Model / Approach':<25} | {'NRMSE (↓)':<15} | {'NMAE (↓)':<15} | {'R² Score (↑)':<15}")
        print("-" * 85)
        print(f"{'Raw Noisy Data':<25} | {nrmse_raw:<15.4f} | {nmae_raw:<15.4f} | {r2_raw:<15.4f}")
        print(f"{'ARIMA Baseline':<25} | {nrmse_ar:<15.4f} | {nmae_ar:<15.4f} | {r2_ar:<15.4f}")
        print(f"{'LSTM Autoencoder':<25} | {nrmse_ls:<15.4f} | {nmae_ls:<15.4f} | {r2_ls:<15.4f}")
        print(f"{'Durrmeyer-Beta (n=80)':<25} | {nrmse_db:<15.4f} | {nmae_db:<15.4f} | {r2_db:<15.4f}")
        print("="*85)