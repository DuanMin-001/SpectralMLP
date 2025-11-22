# visualize_preprocessing.py
# Purpose: Generates Figure 1 for the manuscript, illustrating the effect of
#          key preprocessing steps (SNV and SNV + SG derivative) on a single spectrum.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os  # 确保 os 已导入
from scipy.signal import savgol_filter

# --- Configuration (Relative Path) ---
# Assumes 'cleaned_soil_data.csv' is in the project root.
DEFAULT_CSV_PATH = "cleaned_soil_data.csv"

# 结果保存路径配置 [新增]
RESULT_ROOT = "results"

# Champion SG Parameters (from optimal experiment configuration)
SG_WINDOW = 15
SG_POLYORDER = 3
SG_DERIV = 1


# --- Core Functions (Copied from data_utils.py) ---
# ... (apply_snv 和 apply_sg_smooth 函数保持不变) ...
def apply_snv(X):
    """
    Applies Standard Normal Variate (SNV) transformation to the spectra.

    Args:
        X (pd.DataFrame or np.ndarray): Spectral data.

    Returns:
        np.ndarray: SNV transformed spectra.
    """
    if isinstance(X, pd.DataFrame): X = X.values
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std[std < 1e-9] = 1e-6  # Prevents division by zero
    return (X - mean) / std


def apply_sg_smooth(X, window=15, polyorder=3, deriv=1):
    """
    Applies Savitzky-Golay (SG) filtering (smoothing or derivation).

    Args:
        X (pd.DataFrame or np.ndarray): Spectral data.
        window (int): The length of the filter window (must be odd).
        polyorder (int): The order of the polynomial used to fit the samples.
        deriv (int): The order of the derivative to compute (0=smoothing, 1=1st derivative).

    Returns:
        np.ndarray: SG filtered/derived spectra.
    """
    if isinstance(X, pd.DataFrame): X = X.values
    return savgol_filter(X, window_length=window, polyorder=polyorder, deriv=deriv, axis=1, mode='interp')


# --- Plotting Main Workflow ---
def plot_preprocessing_effect(sample_index=10):
    """
    Loads one sample and plots the effect of three preprocessing stages (Raw, SNV, SNV+SG).

    Args:
        sample_index (int): Index of the sample to visualize.
    """
    print(f"Loading data from: {DEFAULT_CSV_PATH}")
    try:
        df = pd.read_csv(DEFAULT_CSV_PATH)
        target_cols = ['pH.in.CaCl2', 'OC', 'N']
        # Isolate spectral data
        X_df = df.drop(columns=target_cols, errors='ignore')
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        return

    # 1. Get a single sample
    sample_raw = X_df.iloc[sample_index].values.reshape(1, -1)

    # 2. Apply SNV
    sample_snv = apply_snv(sample_raw)

    # 3. Apply SNV + SG Derivative
    sample_snv_sg = apply_sg_smooth(
        sample_snv,
        window=SG_WINDOW,
        polyorder=SG_POLYORDER,
        deriv=SG_DERIV
    )

    # --- Plotting Setup ---
    print("Plotting Figure 1 (Preprocessing Effect)...")

    # Determine x-axis (Wavelengths or Index)
    try:
        wavelengths = pd.to_numeric(X_df.columns)
        x_label = "Wavelength (nm)"
    except:
        wavelengths = np.arange(sample_raw.shape[1])
        x_label = "Wavelength Band Index"

    plt.figure(figsize=(10, 8))

    # Subplot 1 (a): Raw Spectrum
    plt.subplot(3, 1, 1)
    plt.plot(wavelengths, sample_raw.flatten(), label="Raw Spectrum", color='blue')
    plt.title("Figure 1: Effect of Spectral Preprocessing Steps on Raw Data")
    plt.legend(loc="upper right")
    plt.ylabel("Reflectance")
    plt.text(0.01, 0.8, 'a', transform=plt.gca().transAxes, fontsize=14, fontweight='bold')

    # Subplot 2 (b): SNV
    plt.subplot(3, 1, 2)
    plt.plot(wavelengths, sample_snv.flatten(), label="After SNV", color='green')
    plt.legend(loc="upper right")
    plt.ylabel("SNV (Normalized)")
    plt.text(0.01, 0.8, 'b', transform=plt.gca().transAxes, fontsize=14, fontweight='bold')

    # Subplot 3 (c): SNV + SG Derivative
    plt.subplot(3, 1, 3)
    plt.plot(wavelengths, sample_snv_sg.flatten(), label=f"After SNV + SG (deriv={SG_DERIV})", color='red')
    plt.legend(loc="upper right")
    plt.ylabel(f"SG Derivative")
    plt.xlabel(x_label)
    plt.text(0.01, 0.8, 'c', transform=plt.gca().transAxes, fontsize=14, fontweight='bold')

    plt.tight_layout()

    # [!!!] 关键修改: 创建目录并设置保存路径
    os.makedirs(RESULT_ROOT, exist_ok=True)
    save_path = os.path.join(RESULT_ROOT, "figure_1_preprocessing_effect.png")

    plt.savefig(save_path, dpi=300)
    print(f"✅ Figure 1 saved to: {save_path}")


if __name__ == "__main__":
    plot_preprocessing_effect(sample_index=10)