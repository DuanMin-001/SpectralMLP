# =================================================================
# K-Fold Cross-Validation Script (V4 - Efficiency and Parameter Comparison)
# -----------------------------------------------------------------
# Purpose:
#   1. Strictly run 5-Fold Cross-Validation on the 80% Train/Val Set.
#   2. Fairly compare SpectralMLP vs SpectralTransformerMLP regarding:
#      * Performance (Metrics)
#      * Efficiency (Time_to_Best)
#      * Lightweightness (Trainable Parameters)
# =================================================================

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings
import json

# --- Import Custom Modules ---
from model import SpectralMLP, SpectralTransformerMLP
from data_utils import get_data_splits
from utils import (
    seed_everything,
    evaluate_metrics,
    save_metrics_df,
    plot_loss_curve,
    scatter_plots
)


# --- [V4 NEW] Helper Function: Calculate Model Parameters ---
def count_parameters(model):
    """Calculates the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- Suppress PyTorch .grad User Warnings ---
warnings.filterwarnings("ignore", "UserWarning: The .grad attribute of a Tensor", module="torch")

# =================================================================
# --- 1. Global Configuration (Paths, Hyperparameters) ---
# =================================================================

# Use relative paths assuming execution from the project root
CSV_PATH = "cleaned_soil_data.csv"
RESULT_ROOT = "results"
RUN_NAME = "train_kfold_mlpVStransV1"
RESULT_DIR = os.path.join(RESULT_ROOT, RUN_NAME)
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
N_SPLITS = 5  # Number of K-Folds
TARGETS = ["pH", "OC", "N"]

# --- Fixed Optimal Preprocessing (from ablation study) ---
PREPROCESS_PARAMS = {
    'preprocess_method': 'snv_sg_smooth',
    'sg_window': 15,
    'sg_polyorder': 3,
    'sg_deriv': 1
}
TEST_SET_RATIO = 0.2

# --- Fixed Training Control Parameters (from ablation study) ---
TRAINING_PARAMS = {
    'EPOCHS': 250,
    'LR': 3e-4,
    'EARLY_STOPPING_PATIENCE': 20,
    'SCHEDULER_PATIENCE': 10,
    'SCHEDULER_FACTOR': 0.2,
}

# --- Model Parameters (based on optimal settings) ---
PARAMS_SPECTRAL_MLP = {
    'model_name': "SpectralMLP",
    'embed_dim': 128,
    'depth': 2,
    'dropout_rate': 0.0,
    'weight_decay': 0.0,
    'loss_function': 'L1Loss',
    'batch_size': 128,
    'lr': 3e-4,
}

PARAMS_SPECTRAL_TRANSFORMER = {
    'model_name': "SpectralTransformerMLP",
    'embed_dim': 128,
    'depth': 2,
    'num_heads': 8,
    'dropout_rate': 0.0,
    'weight_decay': 0.0,
    'loss_function': 'L1Loss',
    'batch_size': 128,
    'lr': 3e-4,
}

MODELS_TO_RUN = {
    "SpectralMLP": PARAMS_SPECTRAL_MLP,
    "SpectralTransformerMLP": PARAMS_SPECTRAL_TRANSFORMER
}

print(f"--- Running K-Fold (N={N_SPLITS}) Efficiency and Parameter Comparison (V4) ---")
print(f"Comparing Models: {list(MODELS_TO_RUN.keys())}")
print(f"Fixed Preprocessing: {PREPROCESS_PARAMS['preprocess_method']}")
print(f"Results will be saved to: {RESULT_DIR}")
print(f"Using device: {DEVICE}")

# =================================================================
# --- 2. Helper Function: Data Loading (Loads 80% Development Set) ---
# =================================================================
try:
    print("\nLoading 80% Train/Val data (and IGNORING 20% Test data)...")
    # Note: The result signature assumes 4 returned values, ignoring target_names if 5 are returned.
    X_train_val_raw, y_train_val_raw, _, _ = get_data_splits(
        CSV_PATH,
        test_size=TEST_SET_RATIO,
        random_state=SEED,
        **PREPROCESS_PARAMS
    )
    print(f"   Loaded Train/Val set (80%) shape: {X_train_val_raw.shape}")

except Exception as e:
    print(f"[ERROR] Failed to load data: {e}")
    exit()


# =================================================================
# --- 3. Helper Function: Model Instantiation and Training ---
# =================================================================

def get_model(model_name, input_dim, output_dim, params):
    """
    Instantiates the model (SpectralMLP or SpectralTransformerMLP) based on name and parameters.

    Args:
        model_name (str): Name of the model to instantiate.
        input_dim (int): Input feature dimension.
        output_dim (int): Output dimension (targets).
        params (dict): Dictionary of model-specific hyperparameters.

    Returns:
        tuple: (model, num_params)
    """
    model = None
    if model_name == "SpectralMLP":
        model = SpectralMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            embed_dim=params['embed_dim'],
            depth=params['depth'],
            dropout=params['dropout_rate']
        ).to(DEVICE)
    elif model_name == "SpectralTransformerMLP":
        model = SpectralTransformerMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            embed_dim=params['embed_dim'],
            depth=params['depth'],
            num_heads=params.get('num_heads', 8),
            dropout=params['dropout_rate']
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # Calculate and log trainable parameters
    num_params = count_parameters(model)
    print(f"   -> Model {model_name} instantiated. Trainable Parameters: {num_params:,}")

    return model, num_params


def train_one_fold(model, train_loader, val_loader, params, save_dir, num_params):
    """
    (Core Training Function) Trains the model for one K-Fold, incorporating early stopping,
    and saves artifacts (loss curve, best weights).

    Args:
        model (nn.Module): Model instance to train.
        train_loader (DataLoader): DataLoader for the training fold.
        val_loader (DataLoader): DataLoader for the validation fold.
        params (dict): Model-specific parameters (for batch_size, lr, etc.).
        save_dir (str): Directory to save artifacts.
        num_params (int): Number of trainable parameters (passed through for logging).

    Returns:
        tuple: (model, time_to_best_s, best_epoch_num, num_params)
    """

    # --- 1. Loss Function and Optimizer ---
    if params['loss_function'] == 'MSELoss':
        criterion = nn.MSELoss()
    elif params['loss_function'] == 'L1Loss':
        criterion = nn.L1Loss()
    else:
        criterion = nn.SmoothL1Loss()  # Default fallback

    optimizer = AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=TRAINING_PARAMS['SCHEDULER_FACTOR'],
                                  patience=TRAINING_PARAMS['SCHEDULER_PATIENCE'])

    # --- 2. Training Loop (with Early Stopping) ---
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    best_epoch_num_log = 0

    start_time = time.time()

    try:
        for epoch in range(1, TRAINING_PARAMS['EPOCHS'] + 1):
            # --- Training Phase ---
            model.train()
            perm = torch.randperm(train_loader.dataset.tensors[0].size(0))
            running_loss = 0.0
            cnt = 0

            for i in range(0, len(perm), params['batch_size']):
                idx = perm[i:i + params['batch_size']]
                xb = train_loader.dataset.tensors[0][idx].to(DEVICE)
                yb = train_loader.dataset.tensors[1][idx].to(DEVICE)

                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)

                # Check for unstable loss (NaN/Inf)
                if torch.isnan(loss) or torch.isinf(loss): continue

                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)
                cnt += xb.size(0)

            train_loss = running_loss / cnt if cnt > 0 else float('nan')
            train_losses.append(train_loss)

            # --- Evaluation Phase ---
            model.eval()
            val_running_loss = 0.0
            val_cnt = 0
            with torch.no_grad():
                for xb_val, yb_val in val_loader:
                    xb_val, yb_val = xb_val.to(DEVICE), yb_val.to(DEVICE)
                    out_val_batch = model(xb_val)
                    loss_val_batch = criterion(out_val_batch, yb_val)
                    if not (torch.isnan(loss_val_batch) or torch.isinf(loss_val_batch)):
                        val_running_loss += loss_val_batch.item() * xb_val.size(0)
                        val_cnt += xb_val.size(0)

            val_loss = val_running_loss / val_cnt if val_cnt > 0 else float('nan')
            val_losses.append(val_loss)

            # --- Early Stopping and Model Saving ---
            if not np.isnan(val_loss):
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = model.state_dict()
                    epochs_no_improve = 0
                    best_epoch_num_log = epoch
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= TRAINING_PARAMS['EARLY_STOPPING_PATIENCE']:
                        print(f"   -> Early stopping at Epoch {epoch}")
                        break
            else:
                # Handling persistent NaN loss
                epochs_no_improve += 1
                if epochs_no_improve >= TRAINING_PARAMS['EARLY_STOPPING_PATIENCE'] * 2:
                    print(f"   -> Stopping due to persistent NaN Val Loss.")
                    break

    except Exception as e:
        print(f"  [ERROR] Fold training loop failed critically: {e}")

    time_to_best_model = time.time() - start_time

    # --- 3. Save Artifacts ---
    plot_save_path = os.path.join(save_dir, "loss_curve.png")
    plot_loss_curve(train_losses, val_losses, save_path=plot_save_path)

    if best_state is not None:
        model_save_path = os.path.join(save_dir, "best_model.pth")
        torch.save(best_state, model_save_path)
        model.load_state_dict(best_state)
    else:
        print("[WARN] No best model was saved. Evaluation might be unreliable.")

    return model, time_to_best_model, best_epoch_num_log, num_params


# =================================================================
# --- 4. K-Fold CV Loop (Iterate Models) ---
# =================================================================
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

all_models_metrics_dfs = []
efficiency_records = []
total_cv_start_time = time.time()

# --- Iterate through Models (SpectralMLP, SpectralTransformerMLP) ---
for model_name, params in MODELS_TO_RUN.items():

    seed_everything(SEED)

    print(f"\n{'=' * 80}")
    print(f"--- Starting K-Fold CV for MODEL: {model_name} ---")
    print(f"{'=' * 80}")

    model_start_time = time.time()

    model_save_dir = os.path.join(RESULT_DIR, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    # --- Iterate through K-Fold Splits ---
    for fold, (train_fold_idx, val_fold_idx) in enumerate(kf.split(X_train_val_raw, y_train_val_raw)):

        fold_start_time = time.time()
        print(f"\n--- Fold {fold + 1}/{N_SPLITS} for {model_name} ---")

        fold_save_dir = os.path.join(model_save_dir, f"Fold_{fold + 1}")
        os.makedirs(fold_save_dir, exist_ok=True)

        # 4a. Split Data for Current Fold
        X_train_fold_raw = X_train_val_raw[train_fold_idx]
        y_train_fold_raw = y_train_val_raw[train_fold_idx]
        X_val_fold_raw = X_train_val_raw[val_fold_idx]
        y_val_fold_raw = y_train_val_raw[val_fold_idx]

        # 4b. Standardization (Fit only on Fold-Train)
        X_scaler = StandardScaler().fit(X_train_fold_raw)
        y_scaler = StandardScaler().fit(y_train_fold_raw)

        X_train_scaled = X_scaler.transform(X_train_fold_raw)
        y_train_scaled = y_scaler.transform(y_train_fold_raw)
        X_val_scaled = X_scaler.transform(X_val_fold_raw)
        y_val_scaled = y_scaler.transform(y_val_fold_raw)

        input_dim = X_train_scaled.shape[1]
        output_dim = y_train_scaled.shape[1]

        # 4c. Create DataLoaders
        train_ds = torch.utils.data.TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train_scaled, dtype=torch.float32)
        )
        val_ds = torch.utils.data.TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32),
            torch.tensor(y_val_scaled, dtype=torch.float32)
        )
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=params['batch_size'] * 2, shuffle=False)

        # 4d. Instantiate Model and Get Parameter Count
        model, num_params = get_model(model_name, input_dim, output_dim, params)

        # 4e. Train Model, record time and parameters
        model, time_to_best, best_epoch, num_params_out = train_one_fold(
            model, train_loader, val_loader, params, fold_save_dir, num_params
        )

        # 4f. Fold Evaluation
        print(f"   -> Evaluating best model (from Epoch {best_epoch})...")
        model.eval()
        fold_preds_list = []
        with torch.no_grad():
            for xb_val, _ in val_loader:
                fold_preds_list.append(model(xb_val.to(DEVICE)).cpu().numpy())

        y_pred_scaled = np.vstack(fold_preds_list)
        y_pred_scaled = np.nan_to_num(y_pred_scaled)

        # Inverse transform predictions to original scale
        y_true_orig = y_val_fold_raw
        y_pred_orig = y_scaler.inverse_transform(y_pred_scaled)

        print(f"   Metrics for {model_name} Fold {fold + 1}:")
        fold_metrics_df = evaluate_metrics(y_true_orig, y_pred_orig, print_metrics=True)
        fold_metrics_df["Model"] = model_name
        fold_metrics_df["Fold"] = fold + 1
        all_models_metrics_dfs.append(fold_metrics_df)

        scatter_plots(y_true_orig, y_pred_orig, save_dir=fold_save_dir)
        metrics_save_path = os.path.join(fold_save_dir, "metrics.csv")
        save_metrics_df(fold_metrics_df, save_path=metrics_save_path)

        # 4g. Record Efficiency and Parameter Count
        efficiency_records.append({
            "Model": model_name,
            "Fold": fold + 1,
            "Time_to_Best(s)": time_to_best,
            "Best_Epoch": best_epoch,
            "Parameters": num_params_out
        })

        fold_elapsed = time.time() - fold_start_time
        print(f"--- Fold {fold + 1} Finished. Time: {fold_elapsed:.1f}s (Time to Best: {time_to_best:.1f}s) ---")

        # Clean up memory
        del model, X_scaler, y_scaler, train_loader, val_loader, train_ds, val_ds
        torch.cuda.empty_cache()

    model_elapsed = time.time() - model_start_time
    print(f"\n--- K-Fold CV for {model_name} COMPLETE. Total Time: {model_elapsed:.1f}s ---")

# =================================================================
# --- 5. Summary Report (Compare All Models) ---
# =================================================================
print(f"\n{'=' * 80}")
print(f"--- ALL MODELS K-FOLD CV FINAL SUMMARY ---")
print(f"{'=' * 80}")
total_cv_time = time.time() - total_cv_start_time
print(f"Total K-Fold Run Time: {total_cv_time:.1f}s")

if not all_models_metrics_dfs:
    print("[ERROR] No metrics were collected from any model. Aborting.")
    exit()

try:
    # --- 1. Metrics Summary (Mean & Std Dev) ---
    final_summary_df = pd.concat(all_models_metrics_dfs)
    final_stats = final_summary_df.groupby(['Model', 'Target'])[['R2', 'RMSE', 'MAE', 'RPD']].agg(['mean', 'std'])


    def format_mean_std(mean, std):
        return f"{mean:.4f} (± {std:.4f})"


    formatted_stats = pd.DataFrame(index=final_stats.index)
    for col in ['R2', 'RMSE', 'MAE', 'RPD']:
        formatted_stats[col] = [
            format_mean_std(mean, std) for mean, std in zip(final_stats[(col, 'mean')], final_stats[(col, 'std')])
        ]

    print("\n--- Aggregated CV Statistics (Mean & Std Dev) ---")
    print(formatted_stats.to_string())

    final_summary_df.to_csv(os.path.join(RESULT_DIR, "kfold_metrics_summary_raw.csv"), index=False)
    final_stats.to_csv(os.path.join(RESULT_DIR, "kfold_metrics_summary_aggregated.csv"))

    # --- 2. Efficiency and Parameter Summary (JSON) ---
    df_eff = pd.DataFrame(efficiency_records)
    efficiency_summary = {}
    for model_name in MODELS_TO_RUN.keys():
        model_times = df_eff[df_eff["Model"] == model_name]["Time_to_Best(s)"]
        model_epochs = df_eff[df_eff["Model"] == model_name]["Best_Epoch"]
        # Get parameter count (we take the first fold's value since parameters don't change)
        model_params = df_eff[df_eff["Model"] == model_name]["Parameters"].iloc[0]

        efficiency_summary[model_name] = {
            "Avg_Time_to_Best(s)": round(model_times.mean(), 2),
            "Std_Time_to_Best(s)": round(model_times.std(), 2),
            "Avg_Best_Epoch": round(model_epochs.mean(), 1),
            "Trainable_Parameters": int(model_params)
        }

    print("\n--- Efficiency & Parameter Summary (V4) ---")
    print(json.dumps(efficiency_summary, indent=2))

    json_save_path = os.path.join(RESULT_DIR, "kfold_efficiency_summary.json")
    with open(json_save_path, "w") as f:
        json.dump(efficiency_summary, f, indent=2)

    print("\n✅ K-Fold Efficiency Comparison Complete.")
    print(f"   All results saved to: {RESULT_DIR}")

except Exception as e:
    print(f"[ERROR] Failed to aggregate final CV statistics: {e}")
    import traceback;

    traceback.print_exc()

print("=" * 80)