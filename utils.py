import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
import random
import os


def plot_loss_curve(train_losses, val_losses, save_path="loss_curve.png"):
    """
    Plots and saves the training and validation loss curve over epochs.

    Args:
        train_losses (list): List of training loss values.
        val_losses (list): List of validation loss values (can contain NaNs).
        save_path (str): File path to save the plot (e.g., 'results/loss_curve.png').
    """
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label='Train Loss')
    # Check if validation losses exist and are not all NaN before plotting
    if val_losses and not all(np.isnan(v) for v in val_losses):
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss')
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   -> Loss curve saved to: {os.path.basename(save_path)}")
    except Exception as e:
        print(f"[WARN] Failed to save loss curve: {e}")
    plt.close()


def scatter_plots(y_true, y_pred, save_dir="."):
    """
    Generates and saves scatter plots comparing true vs. predicted values
    for the three target soil properties (pH, OC, N).

    Args:
        y_true (np.ndarray): True target values (N samples x 3 targets).
        y_pred (np.ndarray): Predicted target values (N samples x 3 targets).
        save_dir (str): Directory where plots will be saved.
    """
    targets = ["pH", "OC", "N"]
    os.makedirs(save_dir, exist_ok=True)

    for i, t in enumerate(targets):
        plt.figure(figsize=(4, 4))
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=15)

        # Dynamically determine 1:1 line boundaries
        min_val = min(np.min(y_true[:, i]), np.min(y_pred[:, i])) * 0.95
        max_val = max(np.max(y_true[:, i]), np.max(y_pred[:, i])) * 1.05

        # Plot the 1:1 line (ideal prediction)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')

        plt.xlabel(f"True {t}")
        plt.ylabel(f"Predicted {t}")
        plt.title(f"{t} Scatter Plot")
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"scatter_{t}.png")
        try:
            plt.savefig(save_path, dpi=300)
            print(f"   -> Scatter plot saved to: {os.path.basename(save_path)}")
        except Exception as e:
            print(f"[WARN] Failed to save scatter plot: {e}")
        plt.close()


def evaluate_metrics(y_true, y_pred, print_metrics=True):
    """
    Calculates R2 (Coefficient of Determination), RMSE, MAE, and RPD metrics
    for each target property.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
        print_metrics (bool): Whether to print the results to the console.

    Returns:
        pd.DataFrame: A DataFrame containing calculated metrics per target.
    """
    targets = ["pH", "OC", "N"]
    metrics = []
    num_targets = len(targets)

    # Input dimension validation
    if y_true.shape[0] != y_pred.shape[0] or y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("Mismatched samples/dimensions in input arrays.")

    # Adjust number of targets if input array columns are less than 3
    if y_true.shape[1] != num_targets or y_pred.shape[1] != num_targets:
        print(f"[WARN] Metrics: Target shape mismatch.")
        num_targets = min(y_true.shape[1], y_pred.shape[1], len(targets))

    for i in range(num_targets):
        t = targets[i]
        yt_col = y_true[:, i]
        yp_col = y_pred[:, i]

        # Robustness check: skip calculation if non-finite values are present
        if not np.all(np.isfinite(yt_col)) or not np.all(np.isfinite(yp_col)):
            print(f"[WARN] Target {t}: Non-finite values detected.")
            r2, rmse, mae, rpd = [np.nan] * 4
        else:
            try:
                r2 = r2_score(yt_col, yp_col)
                rmse = np.sqrt(mean_squared_error(yt_col, yp_col))
                mae = mean_absolute_error(yt_col, yp_col)
                std_true = np.std(yt_col)

                # RPD (Residual Prediction Deviation) = STD_True / RMSE
                rpd = (std_true / rmse) if rmse > 1e-9 else np.inf

            except Exception as e:
                print(f"[WARN] Target {t}: Error calculating metrics: {e}")
                r2, rmse, mae, rpd = [np.nan] * 4
        metrics.append([t, r2, rmse, mae, rpd])

    df = pd.DataFrame(metrics, columns=["Target", "R2", "RMSE", "MAE", "RPD"])
    if print_metrics:
        # Print metrics without index
        print(df.round(4).to_string(index=False))
    return df


def save_metrics_df(df_metrics, save_path):
    """
    Saves the metrics DataFrame (R2, RMSE, MAE, RPD) to a CSV file.
    Ensures the output directory exists and uses high floating-point precision.

    Args:
        df_metrics (pd.DataFrame): DataFrame containing the evaluation metrics.
        save_path (str): Full file path where the CSV should be saved.
    """
    try:
        dir_name = os.path.dirname(save_path)
        os.makedirs(dir_name, exist_ok=True)
        df_metrics.to_csv(save_path, index=False, float_format='%.6f')
        print(f"✅ Metrics saved to: {os.path.basename(save_path)}")
    except Exception as e:
        print(f"[ERROR] Failed to save metrics: {e}")


def seed_everything(seed=42):
    """
    Sets the global random seed across all relevant libraries (Python, NumPy, PyTorch)
    to ensure full reproducibility of experiments, including CUDA operations.

    Args:
        seed (int): The integer seed value to use.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA configuration for deterministic GPU operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # --- CRITICAL FIX FOR CNN REPRODUCIBILITY ---
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # ---------------------------------------------
    print(f"Set random seed to {seed}")