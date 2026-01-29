# ===== Standard Library =====
import sys
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

# ===== Third-Party Libraries =====
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QListWidget, QTextEdit, QMessageBox, QTabWidget, QCheckBox, QLabel, QDoubleSpinBox,
    QDialog, QFormLayout
)
import re


# ========= USER SETTINGS =========
# Peak positions and indices - hard-coded at top of file
target_peaks = [178.5, 176.8, 170, 160, 124]
target_peaks_labels = [f'P{p}' for p in target_peaks]
method = 'integrals'  # 'integrals' or 'heights'

# Peak index assignments
substratepeak = 2   # Pyruvate (170 ppm)
CO2peak = 4         # CO2 (124 ppm)
HCO3peak = 3        # Bicarbonate (160 ppm)

smoothing = False
startPoint = 2
x_display_min = -5
x_display_max = 400


# ========= Dataclasses =========
@dataclass
class BicarbonateFitConfig:
    """Configuration for the three-compartment Pyruvate → CO₂ ↔ HCO₃⁻ model."""
    
    # Pyruvate → CO₂ production rate bounds
    ksubCO_bounds: Tuple[float, float] = (0.00001, 0.1)
    
    # CO₂ ↔ HCO₃⁻ interconversion rate bounds
    kCOBC_bounds: Tuple[float, float] = (0.001, 0.1)  # CO2 → BIC (hydration)
    kBCCO_bounds: Tuple[float, float] = (0.001, 0.1)  # BIC → CO2 (dehydration)
    
    # T1 relaxation bounds (seconds)
    T1p_bounds_s: Tuple[float, float] = (20.0, 120.0)
    T1CO_bounds_s: Tuple[float, float] = (10.0, 60.0)
    T1BC_bounds_s: Tuple[float, float] = (10.0, 60.0)
    
    # Constraint: T1CO == T1BC
    constrain_T1_CO_BC: bool = True
    
    # Injection time settings
    fit_tinj: bool = False
    fixed_tinj_value: float = 8.0
    tinj_extra_hi: float = 10.0
    
    # AUC integration window (seconds)
    auc_tmin_s: float = 10.0
    auc_tmax_s: float = 90.0


CFG = BicarbonateFitConfig()


@dataclass
class FitResult:
    """Container for fitted parameters and fit status."""
    P0: float           # Initial pyruvate amplitude
    ksubCO: float       # Pyruvate → CO₂ rate
    kCOBC: float        # CO₂ → HCO₃⁻ rate (hydration)
    kBCCO: float        # HCO₃⁻ → CO₂ rate (dehydration)
    Rp_eff: float       # Effective pyruvate decay rate (ksubCO + 1/T1p)
    R_CO2: float        # CO₂ relaxation rate (1/T1CO)
    R_BC: float         # Bicarbonate relaxation rate (1/T1BC)
    tinj: float         # Injection time offset
    success: bool       # Fit convergence status
    message: str        # Fit message


# ========= Smoothing Helper =========
def custom_savgol_filter(x, window_length, polyorder, isolation_range=1, ignore_last=3):
    """
    Apply a Savitzky-Golay filter while ignoring isolated zeros.
    """
    x = np.array(x, dtype=float)
    x_mod = x.copy()
    n = len(x)
    
    for i in range(n):
        if x[i] == 0:
            start = max(0, i - isolation_range)
            end = min(n, i + isolation_range + 1)
            neighbor_indices = [j for j in range(start, end) if j != i]
            if not any(x[j] == 0 for j in neighbor_indices):
                neighbors = [x[j] for j in neighbor_indices if x[j] != 0]
                if neighbors:
                    x_mod[i] = np.mean(neighbors)

    nonzero_indices = np.where(x_mod != 0)[0]
    if len(nonzero_indices) == 0:
        return savgol_filter(x_mod, window_length=window_length, polyorder=polyorder, mode='mirror')

    M = len(nonzero_indices)
    if ignore_last >= M:
        return x_mod

    smooth_end = nonzero_indices[M - ignore_last] + 1
    if smooth_end < window_length:
        smoothed = x_mod
    else:
        smoothed_main = savgol_filter(x_mod[:smooth_end], window_length=window_length, polyorder=polyorder, mode='mirror')
        smoothed = np.concatenate([smoothed_main, x_mod[smooth_end:]])
    smoothed[smoothed < 0] = 0
    return smoothed


# ========= Data Import Functions =========
def import_new_peak_data(csv_file, target_peaks, use_smoothing=False, window_length=5, polyorder=2, isolation_range=1, ignore_last=3):
    """
    Imports peak data from a CSV file.
    Returns array of shape (num_targets, num_rows, 4): [Time, PeakPos, PeakHeight, PeakIntegral]
    """
    df = pd.read_csv(csv_file, skiprows=3, header=None)
    num_targets = len(target_peaks)
    height_cols = list(range(1, 1 + num_targets))
    integral_cols = list(range(1 + num_targets, 1 + 2 * num_targets))

    if use_smoothing:
        for col in height_cols + integral_cols:
            df.iloc[:, col] = custom_savgol_filter(
                df.iloc[:, col].values, window_length, polyorder, isolation_range, ignore_last
            )
    
    df = df.iloc[:150, :]  # Cap at 150 rows
    time_values = df.iloc[:, 0].values

    all_results = []
    for i in range(num_targets):
        height_col = df.iloc[:, i + 1].values
        integral_col = df.iloc[:, i + 1 + num_targets].values
        peakpos_col = np.full_like(time_values, target_peaks[i], dtype=float)
        processed_array = np.column_stack((time_values, peakpos_col, height_col, integral_col))
        all_results.append(processed_array)
    return np.array(all_results)


def create_filtered_arrays(processed_data_array, target_peaks) -> Dict[str, np.ndarray]:
    """Filter arrays to only include rows with positive signal values."""
    filtered_data: Dict[str, np.ndarray] = {}
    for i, processed_array in enumerate(processed_data_array):
        peak_label = f"P{target_peaks[i]}"
        if method == 'heights':
            valid = processed_array[:, 2] > 0
            filtered_data[peak_label] = processed_array[valid]
        elif method == 'integrals':
            valid = processed_array[:, 3] > 0
            filtered_data[peak_label] = processed_array[valid]
        else:
            raise ValueError("method must be 'integrals' or 'heights'")
    return filtered_data


def extract_filtered_data(filtered_arrays, peak_label, start_index=0) -> Tuple[np.ndarray, np.ndarray]:
    """Extract time and signal arrays for a given peak label."""
    if peak_label not in filtered_arrays:
        raise ValueError(f"Peak '{peak_label}' not found.")
    arr = filtered_arrays[peak_label]
    t = np.array(arr[start_index:, 0], dtype=np.float64)
    if method == 'heights':
        y = np.array(arr[start_index:, 2], dtype=np.float64)
    else:
        y = np.array(arr[start_index:, 3], dtype=np.float64)
    return t, y


# ========= Model Functions =========
def _safe_exp(x):
    """Prevent overflow in exp for extreme values."""
    x = np.clip(x, -700, 700)
    return np.exp(x)


def pyruvate_closed_form(t: np.ndarray, P0: float, Rp_eff: float, tinj: float) -> np.ndarray:
    """
    Closed-form solution for pyruvate decay.
    P(t) = P0 * exp(-Rp_eff * (t + tinj))
    where Rp_eff = ksubCO + 1/T1p
    """
    teff = np.asarray(t) + tinj
    teff = np.clip(teff, 0.0, None)  # Enforce non-negative physical time
    return P0 * _safe_exp(-Rp_eff * teff)


def ode_co2_bicarbonate(t, y, params, P_interp):
    """
    ODE system for CO₂ and bicarbonate with pyruvate as forcing function.
    
    y = [CO2, BIC]
    params = [ksubCO, kCOBC, kBCCO, R_CO2, R_BC]
    
    dCO2/dt = ksubCO * P(t) - kCOBC * CO2 + kBCCO * BIC - R_CO2 * CO2
    dBIC/dt = kCOBC * CO2 - kBCCO * BIC - R_BC * BIC
    """
    CO2, BIC = y
    ksubCO, kCOBC, kBCCO, R_CO2, R_BC = params
    
    P_val = P_interp(t)
    
    dCO2 = ksubCO * P_val - kCOBC * CO2 + kBCCO * BIC - R_CO2 * CO2
    dBIC = kCOBC * CO2 - kBCCO * BIC - R_BC * BIC
    
    return [dCO2, dBIC]


def integrate_co2_bicarbonate(params, t_eval, P_interp):
    """
    Integrate the CO₂/bicarbonate ODE system.
    
    params = [ksubCO, kCOBC, kBCCO, R_CO2, R_BC]
    
    Returns interpolated CO2 and BIC values at t_eval.
    """
    t_start = max(0, t_eval[0])
    t_end = t_eval[-1]
    
    # Dense integration grid
    t_dense = np.linspace(t_start, t_end, 500)
    
    sol = solve_ivp(
        lambda t, y: ode_co2_bicarbonate(t, y, params, P_interp),
        t_span=(t_start, t_end),
        y0=[0.0, 0.0],  # Initial CO2 and BIC are zero
        t_eval=t_dense,
        method='RK45'
    )
    
    # Interpolate to requested time points
    CO2_interp = interp1d(t_dense, sol.y[0], kind='linear', bounds_error=False, fill_value="extrapolate")
    BIC_interp = interp1d(t_dense, sol.y[1], kind='linear', bounds_error=False, fill_value="extrapolate")
    
    return CO2_interp(t_eval), BIC_interp(t_eval)


# ========= Residual and Fitting Functions =========
def _build_residuals(theta, tP, Pobs, tCO2, CO2obs, tBC, BCobs, cfg: BicarbonateFitConfig, tinj: float):
    """
    Build residual vector for joint fitting of Pyruvate, CO₂, and Bicarbonate.
    
    theta contents depend on cfg.constrain_T1_CO_BC:
        If constrained: [P0, ksubCO, kCOBC, kBCCO, Rp_eff, R_CO2_BC]
        If independent:  [P0, ksubCO, kCOBC, kBCCO, Rp_eff, R_CO2, R_BC]
    """
    if cfg.constrain_T1_CO_BC:
        P0, ksubCO, kCOBC, kBCCO, Rp_eff, R_CO2_BC = theta
        R_CO2 = R_CO2_BC
        R_BC = R_CO2_BC
    else:
        P0, ksubCO, kCOBC, kBCCO, Rp_eff, R_CO2, R_BC = theta
    
    # Compute pyruvate model (closed-form)
    Pmod = pyruvate_closed_form(tP, P0, Rp_eff, tinj)
    
    # Create pyruvate interpolator for ODE forcing
    # Use a dense time grid for smooth interpolation
    t_all = np.unique(np.concatenate([tP, tCO2, tBC]))
    t_dense_for_interp = np.linspace(0, max(t_all.max(), tP.max()) + 10, 500)
    P_dense = pyruvate_closed_form(t_dense_for_interp, P0, Rp_eff, tinj)
    P_interp = interp1d(t_dense_for_interp, P_dense, kind='linear', bounds_error=False, fill_value=0.0)
    
    # Compute CO2 and bicarbonate models (ODE integration)
    ode_params = [ksubCO, kCOBC, kBCCO, R_CO2, R_BC]
    
    # Evaluate at CO2 time points
    CO2mod, _ = integrate_co2_bicarbonate(ode_params, tCO2, P_interp)
    
    # Evaluate at bicarbonate time points
    _, BCmod = integrate_co2_bicarbonate(ode_params, tBC, P_interp)
    
    # Normalize residuals by max observed values
    wp = 1.0 / max(np.max(Pobs), 1e-9)
    wco2 = 1.0 / max(np.max(CO2obs), 1e-9)
    wbc = 1.0 / max(np.max(BCobs), 1e-9)
    
    rP = wp * (Pmod - Pobs)
    rCO2 = wco2 * (CO2mod - CO2obs)
    rBC = wbc * (BCmod - BCobs)
    
    return np.concatenate([rP, rCO2, rBC], axis=0)


def fit_three_compartment(tP_raw, Pobs_raw, tCO2_raw, CO2obs_raw, tBC_raw, BCobs_raw, 
                          cfg: BicarbonateFitConfig) -> FitResult:
    """
    Fit the three-compartment model: Pyruvate → CO₂ ↔ HCO₃⁻
    
    Returns FitResult with all fitted parameters.
    """
    # Time shift: reference to first pyruvate point
    t0 = float(tP_raw[0])
    tP = tP_raw - t0
    tCO2 = tCO2_raw - t0
    tBC = tBC_raw - t0
    
    # --- tinj setup ---
    tinj0_guess = max(0.0, t0 + 5.0)
    tinj_lo = max(0.0, t0)
    tinj_hi = t0 + cfg.tinj_extra_hi
    tinj_for_residuals = tinj0_guess if cfg.fit_tinj else float(cfg.fixed_tinj_value)
    
    # --- Convert T1 bounds to rate bounds ---
    # R = 1/T1, so R_lo = 1/T1_hi and R_hi = 1/T1_lo
    Rp_lo = 1.0 / cfg.T1p_bounds_s[1]
    Rp_hi = 1.0 / cfg.T1p_bounds_s[0]
    R_CO2_lo = 1.0 / cfg.T1CO_bounds_s[1]
    R_CO2_hi = 1.0 / cfg.T1CO_bounds_s[0]
    R_BC_lo = 1.0 / cfg.T1BC_bounds_s[1]
    R_BC_hi = 1.0 / cfg.T1BC_bounds_s[0]
    
    # --- Initial guesses ---
    P0_guess = max(float(np.max(Pobs_raw)), 1e-6)
    
    # Estimate Rp_eff from early pyruvate decay
    early = min(len(tP), 8)
    if early >= 3 and np.all(Pobs_raw[:early] > 0):
        coeffs = np.polyfit(tP[:early], np.log(Pobs_raw[:early]), 1)
        Rp_eff_guess = float(np.clip(-coeffs[0], Rp_lo + cfg.ksubCO_bounds[0], Rp_hi + cfg.ksubCO_bounds[1]))
    else:
        Rp_eff_guess = 0.02
    
    ksubCO_guess = 0.005
    kCOBC_guess = 0.01
    kBCCO_guess = 0.01
    R_CO2_guess = 0.05  # ~20s T1
    R_BC_guess = 0.05
    
    # --- Build parameter vector and bounds ---
    if cfg.constrain_T1_CO_BC:
        # Constrained: R_CO2 == R_BC
        x0 = np.array([P0_guess, ksubCO_guess, kCOBC_guess, kBCCO_guess, Rp_eff_guess, R_CO2_guess])
        
        lb = np.array([0.0, cfg.ksubCO_bounds[0], cfg.kCOBC_bounds[0], cfg.kBCCO_bounds[0],
                       Rp_lo + cfg.ksubCO_bounds[0], min(R_CO2_lo, R_BC_lo)])
        ub = np.array([np.inf, cfg.ksubCO_bounds[1], cfg.kCOBC_bounds[1], cfg.kBCCO_bounds[1],
                       Rp_hi + cfg.ksubCO_bounds[1], max(R_CO2_hi, R_BC_hi)])
        
        if cfg.fit_tinj:
            x0 = np.append(x0, tinj0_guess)
            lb = np.append(lb, tinj_lo)
            ub = np.append(ub, tinj_hi)
            
            def fun_fit(theta):
                params = theta[:-1]
                tinj_ = theta[-1]
                return _build_residuals(params, tP, Pobs_raw, tCO2, CO2obs_raw, tBC, BCobs_raw, cfg, tinj_)
        else:
            def fun_fit(theta):
                return _build_residuals(theta, tP, Pobs_raw, tCO2, CO2obs_raw, tBC, BCobs_raw, cfg, tinj_for_residuals)
        
        res = least_squares(fun_fit, x0, bounds=(lb, ub), method='trf', max_nfev=10000)
        
        if cfg.fit_tinj:
            P0, ksubCO, kCOBC, kBCCO, Rp_eff, R_CO2_BC, tinj = res.x
        else:
            P0, ksubCO, kCOBC, kBCCO, Rp_eff, R_CO2_BC = res.x
            tinj = tinj_for_residuals
        
        R_CO2 = R_CO2_BC
        R_BC = R_CO2_BC
        
    else:
        # Independent: R_CO2 and R_BC fitted separately
        x0 = np.array([P0_guess, ksubCO_guess, kCOBC_guess, kBCCO_guess, Rp_eff_guess, R_CO2_guess, R_BC_guess])
        
        lb = np.array([0.0, cfg.ksubCO_bounds[0], cfg.kCOBC_bounds[0], cfg.kBCCO_bounds[0],
                       Rp_lo + cfg.ksubCO_bounds[0], R_CO2_lo, R_BC_lo])
        ub = np.array([np.inf, cfg.ksubCO_bounds[1], cfg.kCOBC_bounds[1], cfg.kBCCO_bounds[1],
                       Rp_hi + cfg.ksubCO_bounds[1], R_CO2_hi, R_BC_hi])
        
        if cfg.fit_tinj:
            x0 = np.append(x0, tinj0_guess)
            lb = np.append(lb, tinj_lo)
            ub = np.append(ub, tinj_hi)
            
            def fun_fit(theta):
                params = theta[:-1]
                tinj_ = theta[-1]
                return _build_residuals(params, tP, Pobs_raw, tCO2, CO2obs_raw, tBC, BCobs_raw, cfg, tinj_)
        else:
            def fun_fit(theta):
                return _build_residuals(theta, tP, Pobs_raw, tCO2, CO2obs_raw, tBC, BCobs_raw, cfg, tinj_for_residuals)
        
        res = least_squares(fun_fit, x0, bounds=(lb, ub), method='trf', max_nfev=10000)
        
        if cfg.fit_tinj:
            P0, ksubCO, kCOBC, kBCCO, Rp_eff, R_CO2, R_BC, tinj = res.x
        else:
            P0, ksubCO, kCOBC, kBCCO, Rp_eff, R_CO2, R_BC = res.x
            tinj = tinj_for_residuals
    
    return FitResult(
        P0=P0,
        ksubCO=ksubCO,
        kCOBC=kCOBC,
        kBCCO=kBCCO,
        Rp_eff=Rp_eff,
        R_CO2=R_CO2,
        R_BC=R_BC,
        tinj=tinj,
        success=res.success,
        message=str(res.message)
    )


# ========= Orchestrator Function =========
def run_fit_on_file(
    csv_path: str,
    *,
    target_peaks: list,
    target_peaks_labels: list,
    substratepeak: int,
    CO2peak: int,
    HCO3peak: int,
    startPoint: int,
    smoothing: bool,
    CFG: BicarbonateFitConfig,
    x_display_min: float,
    x_display_max: float,
) -> dict:
    """
    Orchestrates complete analysis for one file.
    
    Returns payload dict with all data needed for plotting and export.
    """
    # --- Load and preprocess ---
    processed = import_new_peak_data(csv_path, target_peaks, use_smoothing=smoothing)
    filtered = create_filtered_arrays(processed, target_peaks)
    
    # --- Extract time series ---
    P_key = target_peaks_labels[substratepeak]
    CO2_key = target_peaks_labels[CO2peak]
    BC_key = target_peaks_labels[HCO3peak]
    
    tP_raw, Pobs_raw = extract_filtered_data(filtered, P_key, start_index=startPoint)
    tCO2_raw, CO2obs_raw = extract_filtered_data(filtered, CO2_key, start_index=startPoint)
    tBC_raw, BCobs_raw = extract_filtered_data(filtered, BC_key, start_index=startPoint)
    
    # --- Run fit ---
    fit = fit_three_compartment(tP_raw, Pobs_raw, tCO2_raw, CO2obs_raw, tBC_raw, BCobs_raw, CFG)
    
    # --- Derived T1 values ---
    # T1p = 1 / (Rp_eff - ksubCO)
    T1p = np.inf
    if fit.Rp_eff > fit.ksubCO:
        T1p = 1.0 / max(fit.Rp_eff - fit.ksubCO, 1e-12)
    
    T1CO = 1.0 / max(fit.R_CO2, 1e-12)
    T1BC = 1.0 / max(fit.R_BC, 1e-12)
    
    # --- Display alignment (t=0 at first pyruvate point) ---
    t0 = float(tP_raw[0])
    tP = tP_raw - t0
    tCO2 = tCO2_raw - t0
    tBC = tBC_raw - t0
    
    # --- Dense time grid for plotting ---
    t_max = max(
        float(np.max(tP)) if tP.size else 0.0,
        float(np.max(tCO2)) if tCO2.size else 0.0,
        float(np.max(tBC)) if tBC.size else 0.0,
        x_display_max
    )
    t_dense = np.linspace(x_display_min, t_max, 600)
    
    # --- Compute model curves on dense grid ---
    Pmod_dense = pyruvate_closed_form(t_dense, fit.P0, fit.Rp_eff, fit.tinj)
    
    # For ODE integration, we need only the positive time portion
    # Create a separate grid for positive times only
    if t_dense[0] < 0:
        neg_mask = t_dense < 0
        t_dense_positive = t_dense[~neg_mask]  # Only positive times
    else:
        neg_mask = np.zeros(len(t_dense), dtype=bool)
        t_dense_positive = t_dense
    
    # Create pyruvate interpolator for ODE (needs to cover full range for interpolation)
    t_interp_grid = np.linspace(0, t_max + 10, 500)
    P_interp = interp1d(t_interp_grid, pyruvate_closed_form(t_interp_grid, fit.P0, fit.Rp_eff, fit.tinj),
                        kind='linear', bounds_error=False, fill_value=0.0)
    
    ode_params = [fit.ksubCO, fit.kCOBC, fit.kBCCO, fit.R_CO2, fit.R_BC]
    CO2mod_positive, BCmod_positive = integrate_co2_bicarbonate(ode_params, t_dense_positive, P_interp)
    
    # Extend to negative times (zeros) if needed
    if np.any(neg_mask):
        CO2mod_dense = np.zeros_like(t_dense)
        BCmod_dense = np.zeros_like(t_dense)
        CO2mod_dense[~neg_mask] = CO2mod_positive
        BCmod_dense[~neg_mask] = BCmod_positive
    else:
        CO2mod_dense = CO2mod_positive
        BCmod_dense = BCmod_positive
    
    # --- BIC/CO2 ratio ---
    eps = 1e-12
    ratio_dense = BCmod_dense / np.maximum(CO2mod_dense, eps)
    
    # --- pH calculations ---
    # pH from time-weighted average ratio
    mask = (t_dense >= 0) & (t_dense <= 300)
    if np.any(mask):
        weighted_avg_ratio = np.trapezoid(ratio_dense[mask], t_dense[mask]) / (t_dense[mask][-1] - t_dense[mask][0])
    else:
        weighted_avg_ratio = np.nan
    
    pH_from_ratio = np.log10(weighted_avg_ratio) + 6.1 if weighted_avg_ratio > 0 else np.nan
    
    # pH from rates (Henderson-Hasselbalch with equilibrium)
    pH_from_rates = np.log10(fit.kBCCO / fit.kCOBC) + 6.1 if fit.kCOBC > 0 else np.nan
    
    # Instantaneous pH over time
    pH_over_time = np.log10(np.maximum(ratio_dense, eps)) + 6.1
    
    # --- Compute residuals on data points ---
    Pmod_data = pyruvate_closed_form(tP, fit.P0, fit.Rp_eff, fit.tinj)
    
    P_interp_data = interp1d(np.clip(tP, 0, None), pyruvate_closed_form(np.clip(tP, 0, None), fit.P0, fit.Rp_eff, fit.tinj),
                              kind='linear', bounds_error=False, fill_value=0.0)
    
    # For CO2 and BC residuals, need to integrate at their specific time points
    tCO2_pos = np.clip(tCO2, 0, None)
    tBC_pos = np.clip(tBC, 0, None)
    
    # Create full interpolator
    t_interp_full = np.linspace(0, max(tCO2.max(), tBC.max(), tP.max()) + 10, 500)
    P_for_ode = pyruvate_closed_form(t_interp_full, fit.P0, fit.Rp_eff, fit.tinj)
    P_interp_full = interp1d(t_interp_full, P_for_ode, kind='linear', bounds_error=False, fill_value=0.0)
    
    CO2mod_data, _ = integrate_co2_bicarbonate(ode_params, tCO2_pos, P_interp_full)
    _, BCmod_data = integrate_co2_bicarbonate(ode_params, tBC_pos, P_interp_full)
    
    wp = 1.0 / max(float(np.max(Pobs_raw)), 1e-9)
    wco2 = 1.0 / max(float(np.max(CO2obs_raw)), 1e-9)
    wbc = 1.0 / max(float(np.max(BCobs_raw)), 1e-9)
    
    rP_norm = wp * (Pmod_data - Pobs_raw)
    rCO2_norm = wco2 * (CO2mod_data - CO2obs_raw)
    rBC_norm = wbc * (BCmod_data - BCobs_raw)
    
    # --- AUC calculations ---
    w_auc = (t_dense >= CFG.auc_tmin_s) & (t_dense <= CFG.auc_tmax_s)
    if np.any(w_auc):
        AUC_CO = float(np.trapezoid(CO2mod_dense[w_auc], t_dense[w_auc]))
        AUC_BC = float(np.trapezoid(BCmod_dense[w_auc], t_dense[w_auc]))
    else:
        AUC_CO = 0.0
        AUC_BC = 0.0
    
    # --- Assemble payload ---
    payload = dict(
        file=Path(csv_path).name,
        # Raw data
        tP_raw=tP, Pobs_raw=Pobs_raw,
        tCO2_raw=tCO2, CO2obs_raw=CO2obs_raw,
        tBC_raw=tBC, BCobs_raw=BCobs_raw,
        # Fit result
        fit=fit,
        # Derived T1 values
        T1p=T1p,
        T1CO=T1CO,
        T1BC=T1BC,
        t0=t0,
        # Dense model curves
        t_dense=t_dense,
        Pmod_dense=Pmod_dense,
        CO2mod_dense=CO2mod_dense,
        BCmod_dense=BCmod_dense,
        ratio_dense=ratio_dense,
        pH_over_time=pH_over_time,
        # pH metrics
        pH_from_ratio=pH_from_ratio,
        pH_from_rates=pH_from_rates,
        weighted_avg_ratio=weighted_avg_ratio,
        # AUC
        AUC_CO=AUC_CO,
        AUC_BC=AUC_BC,
        # Residuals
        rP_norm=rP_norm,
        rCO2_norm=rCO2_norm,
        rBC_norm=rBC_norm,
        # Meta
        x_display_min=float(x_display_min),
        x_display_max=float(x_display_max),
    )
    return payload


# ========= Plotting Functions =========
def _fig_2x3_from_payload(payload, y_max_p=None, y_max_co2=None, y_max_bc=None):
    """Build 2×3 figure from payload."""
    tP = payload["tP_raw"]
    Pobs = payload["Pobs_raw"]
    tCO2 = payload["tCO2_raw"]
    CO2obs = payload["CO2obs_raw"]
    tBC = payload["tBC_raw"]
    BCobs = payload["BCobs_raw"]
    t_dense = payload["t_dense"]
    Pmod_dense = payload["Pmod_dense"]
    CO2mod_dense = payload["CO2mod_dense"]
    BCmod_dense = payload["BCmod_dense"]
    ratio_dense = payload["ratio_dense"]
    pH_over_time = payload["pH_over_time"]
    rP = payload["rP_norm"]
    rCO2 = payload["rCO2_norm"]
    rBC = payload["rBC_norm"]
    xmin = payload["x_display_min"]
    xmax = payload["x_display_max"]
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    axP, axCO2, axBC = axs[0]
    axRatio, axpH, axResid = axs[1]
    
    # --- Pyruvate ---
    axP.scatter(tP, Pobs, s=18, alpha=0.8, color='red', label='Pyruvate (data)')
    axP.plot(t_dense, Pmod_dense, lw=2.0, color='darkred', label='Pyruvate (model)')
    axP.set_xlim(xmin, xmax)
    axP.set_xlabel('Time (s)')
    axP.set_ylabel('Signal (a.u.)')
    if y_max_p is not None:
        axP.set_ylim(-float(y_max_p) / 20, float(y_max_p))
    axP.set_title('Pyruvate Decay')
    axP.legend()
    
    # --- CO₂ ---
    axCO2.scatter(tCO2, CO2obs, s=18, alpha=0.8, color='darkorange', label='CO₂ (data)')
    axCO2.plot(t_dense, CO2mod_dense, lw=2.0, color='orangered', label='CO₂ (model)')
    axCO2.set_xlim(xmin, xmax)
    axCO2.set_xlabel('Time (s)')
    axCO2.set_ylabel('Signal (a.u.)')
    if y_max_co2 is not None:
        axCO2.set_ylim(-float(y_max_co2) / 20, float(y_max_co2))
    axCO2.set_title('CO₂ Accumulation')
    axCO2.legend()
    
    # --- Bicarbonate ---
    axBC.scatter(tBC, BCobs, s=18, alpha=0.8, color='purple', label='HCO₃⁻ (data)')
    axBC.plot(t_dense, BCmod_dense, lw=2.0, color='darkviolet', label='HCO₃⁻ (model)')
    axBC.set_xlim(xmin, xmax)
    axBC.set_xlabel('Time (s)')
    axBC.set_ylabel('Signal (a.u.)')
    if y_max_bc is not None:
        axBC.set_ylim(-float(y_max_bc) / 20, float(y_max_bc))
    axBC.set_title('Bicarbonate Accumulation')
    axBC.legend()
    
    # --- BIC/CO₂ Ratio ---
    axRatio.plot(t_dense, ratio_dense, lw=2.0, color='teal')
    axRatio.set_xlim(xmin, xmax)
    axRatio.set_ylim(0, 3.0)
    axRatio.set_xlabel('Time (s)')
    axRatio.set_ylabel('HCO₃⁻ / CO₂')
    axRatio.set_title('HCO₃⁻/CO₂ Ratio')
    axRatio.grid(True, alpha=0.3)
    
    # --- pH over time ---
    axpH.plot(t_dense, pH_over_time, lw=2.0, color='darkgreen')
    axpH.set_xlim(xmin, xmax)
    axpH.set_ylim(5.0, 8.0)
    axpH.set_xlabel('Time (s)')
    axpH.set_ylabel('Estimated pH')
    axpH.set_title('Estimated pH Over Time')
    axpH.grid(True, alpha=0.3)
    
    # Add pH annotations
    pH_ratio = payload["pH_from_ratio"]
    pH_rates = payload["pH_from_rates"]
    if np.isfinite(pH_ratio):
        axpH.axhline(pH_ratio, ls='--', lw=1, color='blue', alpha=0.7)
        axpH.annotate(f"pH (ratio avg) = {pH_ratio:.2f}", xy=(xmax * 0.6, pH_ratio), fontsize=9, color='blue')
    if np.isfinite(pH_rates):
        axpH.axhline(pH_rates, ls=':', lw=1, color='red', alpha=0.7)
        axpH.annotate(f"pH (rates) = {pH_rates:.2f}", xy=(xmax * 0.6, pH_rates - 0.15), fontsize=9, color='red')
    
    # --- Residuals ---
    axResid.axhline(0, color='k', lw=1)
    axResid.plot(tP, rP, '.', ms=6, label='Pyruvate', color='red')
    axResid.plot(tCO2, rCO2, '.', ms=6, label='CO₂', color='darkorange')
    axResid.plot(tBC, rBC, '.', ms=6, label='HCO₃⁻', color='purple')
    axResid.set_xlim(xmin, xmax)
    axResid.set_xlabel('Time (s)')
    axResid.set_ylabel('Norm. Residual')
    axResid.set_title('Residuals (Normalized)')
    axResid.legend()
    
    fig.tight_layout()
    return fig


def _result_row_from_payload(payload, cfg: BicarbonateFitConfig) -> dict:
    """Flatten key metrics into a dict row for CSV export."""
    fit = payload["fit"]
    row = {
        "file": payload["file"],
        "P0": fit.P0,
        "ksubCO": fit.ksubCO,
        "kCOBC": fit.kCOBC,
        "kBCCO": fit.kBCCO,
        "Rp_eff": fit.Rp_eff,
        "R_CO2": fit.R_CO2,
        "R_BC": fit.R_BC,
        "tinj": fit.tinj,
        "T1p": payload["T1p"],
        "T1CO": payload["T1CO"],
        "T1BC": payload["T1BC"],
        "pH_from_ratio": payload["pH_from_ratio"],
        "pH_from_rates": payload["pH_from_rates"],
        "BIC_CO2_ratio_avg": payload["weighted_avg_ratio"],
        "AUC_CO": payload["AUC_CO"],
        "AUC_BC": payload["AUC_BC"],
        "success": fit.success,
        "message": fit.message,
    }
    return row


# ========= Parameters Dialog =========
class ParametersDialog(QDialog):
    def __init__(self, parent, cfg: BicarbonateFitConfig):
        super().__init__(parent)
        self.setWindowTitle("Fit Parameters (Bounds)")
        self.cfg = cfg
        
        lay = QVBoxLayout(self)
        form = QFormLayout()
        lay.addLayout(form)
        
        # Helper to make min/max spinbox pair
        def mk_pair(lo, hi, step=0.001, decimals=6):
            w_lo = QDoubleSpinBox()
            w_hi = QDoubleSpinBox()
            w_lo.setDecimals(decimals)
            w_hi.setDecimals(decimals)
            w_lo.setRange(-1e9, 1e9)
            w_hi.setRange(-1e9, 1e9)
            w_lo.setSingleStep(step)
            w_hi.setSingleStep(step)
            w_lo.setValue(lo)
            w_hi.setValue(hi)
            row = QHBoxLayout()
            row.addWidget(w_lo)
            row.addWidget(QLabel("to"))
            row.addWidget(w_hi)
            return row, w_lo, w_hi
        
        # ksubCO bounds
        row, self.ksubCO_lo, self.ksubCO_hi = mk_pair(cfg.ksubCO_bounds[0], cfg.ksubCO_bounds[1], 0.00001)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow(QLabel("ksubCO bounds:"), holder)
        
        # kCOBC bounds
        row, self.kCOBC_lo, self.kCOBC_hi = mk_pair(cfg.kCOBC_bounds[0], cfg.kCOBC_bounds[1], 0.001)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow(QLabel("kCOBC bounds (CO₂→HCO₃⁻):"), holder)
        
        # kBCCO bounds
        row, self.kBCCO_lo, self.kBCCO_hi = mk_pair(cfg.kBCCO_bounds[0], cfg.kBCCO_bounds[1], 0.001)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow(QLabel("kBCCO bounds (HCO₃⁻→CO₂):"), holder)
        
        # T1p bounds
        row, self.T1p_lo, self.T1p_hi = mk_pair(cfg.T1p_bounds_s[0], cfg.T1p_bounds_s[1], 1.0, 2)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow(QLabel("T1p bounds (s):"), holder)
        
        # T1CO bounds
        row, self.T1CO_lo, self.T1CO_hi = mk_pair(cfg.T1CO_bounds_s[0], cfg.T1CO_bounds_s[1], 1.0, 2)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow(QLabel("T1CO bounds (s):"), holder)
        
        # T1BC bounds
        row, self.T1BC_lo, self.T1BC_hi = mk_pair(cfg.T1BC_bounds_s[0], cfg.T1BC_bounds_s[1], 1.0, 2)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow(QLabel("T1BC bounds (s):"), holder)
        
        # Constraint checkbox for T1CO == T1BC
        self.cb_constrain_T1 = QCheckBox("Constrain T1CO = T1BC")
        self.cb_constrain_T1.setChecked(cfg.constrain_T1_CO_BC)
        form.addRow(self.cb_constrain_T1)
        
        # AUC window
        self.auc_min = QDoubleSpinBox()
        self.auc_max = QDoubleSpinBox()
        for w in (self.auc_min, self.auc_max):
            w.setDecimals(1)
            w.setRange(-1e6, 1e6)
            w.setSingleStep(5.0)
        self.auc_min.setValue(cfg.auc_tmin_s)
        self.auc_max.setValue(cfg.auc_tmax_s)
        row_auc = QHBoxLayout()
        row_auc.addWidget(self.auc_min)
        row_auc.addWidget(QLabel("to"))
        row_auc.addWidget(self.auc_max)
        holder_auc = QWidget()
        holder_auc.setLayout(row_auc)
        form.addRow(QLabel("AUC window (s):"), holder_auc)
        
        # Buttons
        btns = QHBoxLayout()
        self.btn_save = QPushButton("Save")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_save.clicked.connect(self.on_save)
        self.btn_cancel.clicked.connect(self.reject)
        btns.addWidget(self.btn_save)
        btns.addWidget(self.btn_cancel)
        lay.addLayout(btns)
    
    def on_save(self):
        # Validate min <= max
        pairs = [
            (self.ksubCO_lo.value(), self.ksubCO_hi.value(), "ksubCO"),
            (self.kCOBC_lo.value(), self.kCOBC_hi.value(), "kCOBC"),
            (self.kBCCO_lo.value(), self.kBCCO_hi.value(), "kBCCO"),
            (self.T1p_lo.value(), self.T1p_hi.value(), "T1p"),
            (self.T1CO_lo.value(), self.T1CO_hi.value(), "T1CO"),
            (self.T1BC_lo.value(), self.T1BC_hi.value(), "T1BC"),
            (self.auc_min.value(), self.auc_max.value(), "AUC window"),
        ]
        for lo, hi, name in pairs:
            if hi < lo:
                QMessageBox.warning(self, "Invalid bounds", f"{name} min must be <= max")
                return
        
        # Apply to cfg
        self.cfg.ksubCO_bounds = (self.ksubCO_lo.value(), self.ksubCO_hi.value())
        self.cfg.kCOBC_bounds = (self.kCOBC_lo.value(), self.kCOBC_hi.value())
        self.cfg.kBCCO_bounds = (self.kBCCO_lo.value(), self.kBCCO_hi.value())
        self.cfg.T1p_bounds_s = (self.T1p_lo.value(), self.T1p_hi.value())
        self.cfg.T1CO_bounds_s = (self.T1CO_lo.value(), self.T1CO_hi.value())
        self.cfg.T1BC_bounds_s = (self.T1BC_lo.value(), self.T1BC_hi.value())
        self.cfg.constrain_T1_CO_BC = self.cb_constrain_T1.isChecked()
        self.cfg.auc_tmin_s = self.auc_min.value()
        self.cfg.auc_tmax_s = self.auc_max.value()
        
        self.accept()


def bicarbonate_config_to_dict(cfg: BicarbonateFitConfig) -> dict:
    """Convert config to dict for export."""
    return {
        "ksubCO_bounds": cfg.ksubCO_bounds,
        "kCOBC_bounds": cfg.kCOBC_bounds,
        "kBCCO_bounds": cfg.kBCCO_bounds,
        "T1p_bounds_s": cfg.T1p_bounds_s,
        "T1CO_bounds_s": cfg.T1CO_bounds_s,
        "T1BC_bounds_s": cfg.T1BC_bounds_s,
        "constrain_T1_CO_BC": cfg.constrain_T1_CO_BC,
        "fit_tinj": cfg.fit_tinj,
        "fixed_tinj_value": cfg.fixed_tinj_value,
        "tinj_extra_hi": cfg.tinj_extra_hi,
        "auc_tmin_s": cfg.auc_tmin_s,
        "auc_tmax_s": cfg.auc_tmax_s,
    }


# ========= Utility Functions =========
def _derive_outstem_from_name(stem: str, tag: str) -> str:
    """Generate output filename stem."""
    if "(" in stem:
        name_part = stem[stem.index("("):]
    else:
        name_part = "_" + stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    return f"{name_part}_{tag}_at{timestamp}"


def extract_exptdate(filename: str) -> Optional[str]:
    """Extract 6-digit experiment date from filename."""
    m = re.search(r"integrated_data_(\d{6})", filename)
    return m.group(1) if m else None


# ========= Main GUI =========
class FitGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bicarbonate/CO₂ Fit GUI")
        self.resize(1200, 900)
        self.output_folder = None
        self.canvases = []
        
        root = QVBoxLayout(self)
        
        # --- Top controls ---
        top = QHBoxLayout()
        self.btn_in = QPushButton("Select Input Folder")
        self.btn_in.clicked.connect(self.on_select_input)
        self.btn_out = QPushButton("Select Output Folder")
        self.btn_out.clicked.connect(self.on_select_output)
        self.btn_params = QPushButton("Parameters")
        self.btn_params.clicked.connect(self.on_open_params)
        
        top.addWidget(self.btn_in)
        top.addWidget(self.btn_out)
        top.addWidget(self.btn_params)
        root.addLayout(top)
        
        # --- Export checkboxes ---
        opts = QHBoxLayout()
        opts.addWidget(QLabel("Exports:"))
        self.cb_png = QCheckBox("Plot PNG")
        self.cb_pdf = QCheckBox("Plot PDF")
        self.cb_csv_each = QCheckBox("Per-file CSV")
        self.cb_csv_summary = QCheckBox("Summary CSV")
        self.cb_png.setChecked(True)
        self.cb_pdf.setChecked(True)
        self.cb_csv_each.setChecked(True)
        self.cb_csv_summary.setChecked(True)
        for cb in (self.cb_png, self.cb_pdf, self.cb_csv_each, self.cb_csv_summary):
            opts.addWidget(cb)
        root.addLayout(opts)
        
        # --- Y-axis max limits ---
        ylims = QHBoxLayout()
        ylims.addWidget(QLabel("Y max (Pyruvate, CO₂, HCO₃⁻):"))
        
        self.sb_ymax_p = QDoubleSpinBox()
        self.sb_ymax_p.setRange(0, 1e12)
        self.sb_ymax_p.setDecimals(1)
        self.sb_ymax_p.setValue(3500.0)
        
        self.sb_ymax_co2 = QDoubleSpinBox()
        self.sb_ymax_co2.setRange(0, 1e12)
        self.sb_ymax_co2.setDecimals(1)
        self.sb_ymax_co2.setValue(50.0)
        
        self.sb_ymax_bc = QDoubleSpinBox()
        self.sb_ymax_bc.setRange(0, 1e12)
        self.sb_ymax_bc.setDecimals(1)
        self.sb_ymax_bc.setValue(50.0)
        
        ylims.addWidget(self.sb_ymax_p)
        ylims.addWidget(self.sb_ymax_co2)
        ylims.addWidget(self.sb_ymax_bc)
        root.addLayout(ylims)
        
        # --- File list ---
        files_row = QHBoxLayout()
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        files_row.addWidget(self.file_list)
        root.addLayout(files_row)
        
        # --- Run button ---
        runrow = QHBoxLayout()
        self.btn_run = QPushButton("Run Fit")
        self.btn_run.clicked.connect(self.on_run_fit)
        runrow.addWidget(self.btn_run)
        root.addLayout(runrow)
        
        # --- Tabs for results and plots ---
        self.tabs = QTabWidget()
        self.output_tab = QTextEdit()
        self.output_tab.setReadOnly(True)
        self.tabs.addTab(self.output_tab, "Results")
        root.addWidget(self.tabs)
    
    def on_open_params(self):
        dlg = ParametersDialog(self, CFG)
        if dlg.exec_() == QDialog.Accepted:
            cfgd = bicarbonate_config_to_dict(CFG)
            self.output_tab.append(
                f"[Parameters] Updated: "
                f"ksubCO={cfgd['ksubCO_bounds']}, "
                f"kCOBC={cfgd['kCOBC_bounds']}, "
                f"kBCCO={cfgd['kBCCO_bounds']}, "
                f"T1p={cfgd['T1p_bounds_s']}, "
                f"T1CO={cfgd['T1CO_bounds_s']}, "
                f"T1BC={cfgd['T1BC_bounds_s']}, "
                f"constrain_T1={cfgd['constrain_T1_CO_BC']}, "
                f"AUC_window=({cfgd['auc_tmin_s']}, {cfgd['auc_tmax_s']})"
            )
    
    def on_select_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if not folder:
            return
        self.file_list.clear()
        for f in os.listdir(folder):
            if f.lower().endswith(".csv"):
                self.file_list.addItem(str(Path(folder) / f))
        self.output_tab.append(f"Input folder set: {folder}")
    
    def on_select_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder:
            return
        self.output_folder = Path(folder)
        self.output_tab.append(f"Output folder set: {self.output_folder}")
    
    def on_run_fit(self):
        selected = [it.text() for it in self.file_list.selectedItems()]
        if not selected:
            QMessageBox.warning(self, "No files", "Please select at least one CSV.")
            return
        if self.output_folder is None and any([self.cb_png.isChecked(), self.cb_pdf.isChecked(),
                                               self.cb_csv_each.isChecked(), self.cb_csv_summary.isChecked()]):
            QMessageBox.warning(self, "No output folder", "Select an output folder for exports.")
            return
        
        # Clear previous tabs (except results)
        while self.tabs.count() > 1:
            self.tabs.removeTab(1)
        self.canvases.clear()
        
        do_png = self.cb_png.isChecked()
        do_pdf = self.cb_pdf.isChecked()
        do_csv_each = self.cb_csv_each.isChecked()
        do_csv_summary = self.cb_csv_summary.isChecked()
        
        all_rows = []
        self.output_tab.append("\nRunning fits...\n")
        
        for file in selected:
            try:
                payload = run_fit_on_file(
                    file,
                    target_peaks=target_peaks,
                    target_peaks_labels=target_peaks_labels,
                    substratepeak=substratepeak,
                    CO2peak=CO2peak,
                    HCO3peak=HCO3peak,
                    startPoint=startPoint,
                    smoothing=smoothing,
                    CFG=CFG,
                    x_display_min=x_display_min,
                    x_display_max=x_display_max,
                )
            except Exception as e:
                self.output_tab.append(f"ERROR processing {Path(file).name}: {e}")
                import traceback
                self.output_tab.append(traceback.format_exc())
                continue
            
            # Print results to GUI
            row = _result_row_from_payload(payload, CFG)
            all_rows.append(row)
            self.output_tab.append(f"=== {row['file']} ===")
            for k, v in row.items():
                if k != "file":
                    if isinstance(v, float):
                        self.output_tab.append(f"  {k}: {v:.6g}")
                    else:
                        self.output_tab.append(f"  {k}: {v}")
            self.output_tab.append("----")
            
            # Create figure and embed as tab
            fig = _fig_2x3_from_payload(
                payload,
                y_max_p=self.sb_ymax_p.value(),
                y_max_co2=self.sb_ymax_co2.value(),
                y_max_bc=self.sb_ymax_bc.value(),
            )
            canvas = FigureCanvas(fig)
            canvas.draw()
            self.tabs.addTab(canvas, Path(file).name)
            self.canvases.append(canvas)
            
            # Exports
            if self.output_folder is not None:
                stem = Path(file).stem
                exptdate = extract_exptdate(stem) or ""
                prefix = f"{exptdate}_" if exptdate else ""
                outstem = f"{prefix}{_derive_outstem_from_name(stem, 'bicfit')}"
                
                if do_csv_each:
                    out_csv = self.output_folder / f"{outstem}.csv"
                    pd.DataFrame([row]).to_csv(out_csv, index=False)
                
                if do_png:
                    out_png = self.output_folder / f"{outstem}.png"
                    fig.savefig(out_png, dpi=300)
                
                if do_pdf:
                    out_pdf = self.output_folder / f"{outstem}.pdf"
                    fig.savefig(out_pdf)
            
            plt.close(fig)
        
        # Summary CSV
        if do_csv_summary and all_rows and self.output_folder is not None:
            first_file = selected[0] if selected else ""
            exptdate = extract_exptdate(os.path.basename(first_file)) or ""
            datepart = f"{exptdate}_" if exptdate else ""
            summary_path = self.output_folder / f"summary_{datepart}bicfit_at{datetime.now().strftime('%Y%m%d-%H%M')}.csv"
            pd.DataFrame(all_rows).to_csv(summary_path, index=False)
            self.output_tab.append(f"\nSummary saved to {summary_path}")
        
        # Write fit parameters file
        if any([do_png, do_pdf, do_csv_each, do_csv_summary]) and self.output_folder is not None:
            first_file = selected[0] if selected else ""
            exptdate = extract_exptdate(os.path.basename(first_file)) or ""
            datepart = f"{exptdate}_" if exptdate else ""
            fitparam_path = self.output_folder / f"fitparameters_{datepart}bicfit_at{datetime.now().strftime('%Y%m%d-%H%M')}.txt"
            try:
                with open(fitparam_path, 'w', encoding='utf-8') as fp:
                    for k, v in bicarbonate_config_to_dict(CFG).items():
                        fp.write(f"{k} = {v}\n")
                self.output_tab.append(f"Fit parameters saved to {fitparam_path}")
            except Exception as e:
                self.output_tab.append(f"ERROR writing fit parameters: {e}")
        
        self.output_tab.append("\nDone.\n")


# ========= Entrypoint =========
def main():
    app = QApplication(sys.argv)
    gui = FitGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
