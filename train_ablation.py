# =================================================================
# train_ablation.py (V3 - With Artifact Saving)
# -----------------------------------------------------------------
# Purpose:
#   1. [!!!] Strictly run the experiment within the 80% Train/Val dataset.
#   2. Split the 80% dataset internally into 64% (Training) / 16% (Validation).
#   3. Train on 64%, use the 16% set for Early Stopping.
#   4. [V3 NEW] Create dedicated folders for each experiment and save artifacts
#      (best_model.pth, loss_curve.png, scatter_plots, metrics.csv).
# =================================================================

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import itertools
import warnings
import re

# --- Import Custom Modules ---
from data_utils import get_data_splits, augment_data
from model import SpectralMLP, SpectralTransformerMLP
from utils import (
    seed_everything,
    evaluate_metrics,
    save_metrics_df,
    plot_loss_curve,
    scatter_plots
)

# --- Suppress PyTorch .grad User Warnings ---
warnings.filterwarnings("ignore", "UserWarning: The .grad attribute of a Tensor", module="torch")

# =================================================================
# --- 1. Global Configuration (Relative Paths) ---
# =================================================================
# Replace absolute paths with relative paths
CSV_PATH = "cleaned_soil_data.csv"
RESULT_ROOT = "results"
RUN_NAME = "train_ablation"  # Ablation experiment result folder
RESULT_DIR = os.path.join(RESULT_ROOT, RUN_NAME)
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(42)

# --- Training Control Parameters (Consistent with final_evaluation.py) ---
EPOCHS = 250
LR = 3e-4
EARLY_STOPPING_PATIENCE = 20
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.2

# --- Data Split Ratios (Critical) ---
TEST_SET_RATIO = 0.2  # Original 80/20 split ratio from get_data_splits
ABALATION_VAL_RATIO = 0.2  # Split 20% from the 80% dev set for the *internal validation set* (16% of total)

print(f"--- Running Rigorous Ablation Experiment (V3 - With Artifact Saving) ---")
print(f"--- [!!!] WARNING: This script will *NEVER* access the 20% independent test set. ---")
print(f"--- Experiment runs on an internal 64%/16% split within the 80% development set. ---")
print(f"Results will be saved to: {RESULT_DIR}")
print(f"Using device: {DEVICE}")

# =================================================================
# --- 2. Champion Model / Control Group Definition ---
# =================================================================
# [!!!] Champion Parameters (Based on V2/V3 log findings):
CHAMPION_PARAMS = {
    'model_name': "SpectralMLP",
    'preprocess_method': 'snv_sg_smooth',
    'sg_window': 15,
    'sg_polyorder': 3,
    'sg_deriv': 1,
    'embed_dim': 128,
    'depth': 2,
    'num_heads': 8,  # (Placeholder for Transformer ABLATION 1, MLP ignores it)
    'dropout_rate': 0.0,
    'weight_decay': 0.0,
    'use_augmentation': False,
    'loss_function': "L1Loss",
    'batch_size': 128,
}

# =================================================================
# --- 3. Ablation Experiment Definitions ---
# =================================================================
# Format: ( "Experiment Name", {Dictionary of Overriding Parameters} )
ABLATON_EXPERIMENTS = [
    # ----------------------------------------------------
    # Baseline (Must be the first experiment)
    ("Baseline (Champion_SpectralMLP)", {}),
    # ----------------------------------------------------

    # Ablation 2 (Preprocessing)
    ("Ablation 2a: Preprocess(Raw)", {'preprocess_method': None}),
    ("Ablation 2b: Preprocess(SNV)", {'preprocess_method': 'snv'}),
    # (Baseline is 'snv_sg_smooth')

    # Ablation 3 (Model Capacity - EmbedDim)
    ("Ablation 3a: EmbedDim(64)", {'embed_dim': 64}),
    ("Ablation 3b: EmbedDim(256)", {'embed_dim': 256}),
    # (Baseline is 128)

    # Ablation 4 (Model Capacity - Depth)
    ("Ablation 4a: Depth(1)", {'depth': 1}),
    ("Ablation 4b: Depth(3)", {'depth': 3}),
    # (Baseline is 2)

    # Ablation 5 (Regularization)
    ("Ablation 5a: Dropout(0.1)", {'dropout_rate': 0.1}),
    ("Ablation 5b: WeightDecay(1e-5)", {'weight_decay': 1e-5}),
    # (Baseline is 0.0)

    # Ablation 6 (Data Augmentation)
    ("Ablation 6: UseAugmentation(True)", {'use_augmentation': True}),
    # (Baseline is False)

    # Ablation 7 (Loss Function)
    ("Ablation 7a: Loss(MSEL1)", {'loss_function': 'MSELoss'}),
    ("Ablation 7b: Loss(SmoothL1)", {'loss_function': 'SmoothL1Loss'}),
    # (Baseline is L1Loss)
]


# =================================================================
# --- 4. Helper Function: Data Loading (With Cache) ---
# =================================================================
def load_and_split_ablation_data(config, data_cache):
    """
    Loads, caches, and splits data.
    [!!!] Strictly operates only within the 80% development set.
    """

    # 1. Determine Data Configuration
    prep_method = config['preprocess_method']
    sg_w = config.get('sg_window', 0)
    sg_p = config.get('sg_polyorder', 0)
    sg_d = config.get('sg_deriv', 0)
    use_aug = config['use_augmentation']

    # 2. Preprocessing Cache Key (Only caches the preprocessing, not augmentation)
    preprocess_key = f"{prep_method}_W{sg_w}_P{sg_p}_D{sg_d}"

    if preprocess_key in data_cache:
        print("   -> (Loading preprocessed data from cache...)")
        (X_train_abl_raw, y_train_abl_raw,
         X_val_abl_raw, y_val_abl_raw) = data_cache[preprocess_key]
    else:
        print(f"   -> (Cache miss. Loading data for: {preprocess_key})")

        # 3. Load 80% (Train/Val) + 20% (Test)
        # We *discard* X_test_raw, y_test_raw to prevent any data leakage
        X_train_val_raw, y_train_val_raw, _, _ = get_data_splits(
            CSV_PATH,
            preprocess_method=prep_method,
            test_size=TEST_SET_RATIO,
            sg_window=sg_w,
            sg_polyorder=sg_p,
            sg_deriv=sg_d
        )
        print(f"   -> Loaded 80% Data: {X_train_val_raw.shape}")

        # 4. Internal Split: 80% -> 64% (Train) / 16% (Val)
        X_train_abl_raw, X_val_abl_raw, y_train_abl_raw, y_val_abl_raw = train_test_split(
            X_train_val_raw, y_train_val_raw, test_size=ABALATION_VAL_RATIO, random_state=42
        )
        print(f"   -> Internal Split: Train (64%): {X_train_abl_raw.shape}, Val (16%): {X_val_abl_raw.shape}")

        # Store in cache
        data_cache[preprocess_key] = (X_train_abl_raw, y_train_abl_raw, X_val_abl_raw, y_val_abl_raw)

    # 5. Apply Data Augmentation (if required)
    if use_aug:
        print("   -> Applying augmentation...")
        X_train_aug, y_train_aug = augment_data(X_train_abl_raw, y_train_abl_raw, random_state=42)
    else:
        X_train_aug, y_train_aug = X_train_abl_raw, y_train_abl_raw

    # 6. Standardization (Fit ONLY on the 64% augmented train set)
    print("   -> Fitting scalers ONLY on 64% train set...")
    X_scaler = StandardScaler().fit(X_train_aug)
    y_scaler = StandardScaler().fit(y_train_aug)

    print("   -> Transforming all data splits...")
    X_train_t = torch.tensor(X_scaler.transform(X_train_aug), dtype=torch.float32)
    y_train_t = torch.tensor(y_scaler.transform(y_train_aug), dtype=torch.float32)
    X_val_t = torch.tensor(X_scaler.transform(X_val_abl_raw), dtype=torch.float32)
    y_val_t = torch.tensor(y_scaler.transform(y_val_abl_raw), dtype=torch.float32)

    input_dim = X_train_t.shape[1]
    output_dim = y_train_t.shape[1]

    # Return all necessary data
    return (X_train_t, y_train_t, X_val_t, y_val_t,
            X_val_abl_raw, y_val_abl_raw,
            y_scaler, input_dim, output_dim)


# =================================================================
# --- 5. Helper Function: Train Single Experiment ---
# =================================================================
def train_single_experiment(exp_name, config, data_package, exp_save_dir):
    """
    (Core Function) Trains and evaluates a single configuration.
    Uses Early Stopping and returns metrics from the *best* model state.

    Args:
        exp_name (str): Name of the current experiment.
        config (dict): Configuration dictionary (model params, etc.).
        data_package (tuple): Pre-loaded and split data tensors/arrays.
        exp_save_dir (str): Directory for saving artifacts.

    Returns:
        dict: Results dictionary for the summary table.
    """

    print("\n" + "-" * 80)
    print(f"--- Running Exp: {exp_name}")
    print(f"    Config: {config}")
    print(f"    Saving artifacts to: {os.path.basename(exp_save_dir)}")
    print("-" * 80)

    # --- 1. Unpack Data ---
    (X_train_t, y_train_t, X_val_t, y_val_t,
     X_val_abl_raw, y_val_abl_raw,
     y_scaler, input_dim, output_dim) = data_package

    # --- 2. Extract Configuration ---
    model_name = config['model_name']
    batch_size = config['batch_size']
    weight_decay = config['weight_decay']

    # --- 3. Instantiate Model (Based on Name) ---
    if model_name == "SpectralMLP":
        model = SpectralMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            dropout=config['dropout_rate']
        ).to(DEVICE)
    elif model_name == "SpectralTransformerMLP":
        model = SpectralTransformerMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            num_heads=config['num_heads'],
            dropout=config['dropout_rate']
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # --- 4. Loss Function and Optimizer ---
    if config['loss_function'] == 'MSELoss':
        criterion = nn.MSELoss()
    elif config['loss_function'] == 'L1Loss':
        criterion = nn.L1Loss()
    else:  # SmoothL1Loss
        criterion = nn.SmoothL1Loss()

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)

    # --- 5. Training Loop (with Early Stopping) ---
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    training_stable = True
    stopped_early = False
    best_epoch_num_log = 0

    print("   Starting FAIR training loop (with Early Stopping)...")
    start_time = time.time()

    try:
        for epoch in range(1, EPOCHS + 1):
            # --- Training Phase (on 64% data) ---
            model.train()
            perm = torch.randperm(X_train_t.size(0))
            running_loss = 0.0;
            cnt = 0;
            train_nan_detected = False

            for i in range(0, len(perm), batch_size):
                idx = perm[i:i + batch_size]
                xb = X_train_t[idx].to(DEVICE);
                yb = y_train_t[idx].to(DEVICE)
                optimizer.zero_grad();
                out = model(xb);
                loss = criterion(out, yb)
                if torch.isnan(loss) or torch.isinf(loss):
                    train_nan_detected = True;
                    continue
                loss.backward();
                optimizer.step()
                running_loss += loss.item() * xb.size(0);
                cnt += xb.size(0)

            if cnt == 0 and train_nan_detected:
                print(f"  [ERROR] Epoch {epoch}: All train batches failed (NaN). Aborting config.");
                training_stable = False;
                break
            train_loss = running_loss / cnt if cnt > 0 else float('nan')
            train_losses.append(train_loss)
            if np.isnan(train_loss):
                print(f"  [ERROR] Epoch {epoch}: Train loss is NaN. Aborting config.");
                training_stable = False;
                break

            # --- Evaluation Phase (on 16% data) ---
            model.eval()
            val_running_loss = 0.0;
            val_cnt = 0
            with torch.no_grad():
                for i_val in range(0, X_val_t.size(0), batch_size):
                    xb_val = X_val_t[i_val:i_val + batch_size].to(DEVICE)
                    yb_val = y_val_t[i_val:i_val + batch_size].to(DEVICE)
                    out_val_batch = model(xb_val);
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
                    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                        print(f"   -> Early stopping at Epoch {epoch} (Best Val Loss: {best_val_loss:.4f})")
                        stopped_early = True;
                        break
            else:
                print(f"  [WARN] Epoch {epoch}: NaN validation loss.")
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE * 2:
                    print(f"   -> Stopping due to persistent NaN Val Loss.");
                    training_stable = False;
                    stopped_early = True;
                    break

            if epoch % 40 == 0:
                print(f"   Epoch [{epoch}/{EPOCHS}] Train Loss: {train_loss:.4f} | VAL LOSS: {val_loss:.4f}")

    except Exception as e:
        print(f"  [ERROR] Training loop failed critically for {exp_name}: {e}")
        import traceback;
        traceback.print_exc();
        training_stable = False

    # Record time to reach the best model
    time_to_best_model = time.time() - start_time

    # --- Save Loss Curve ---
    plot_save_path = os.path.join(exp_save_dir, "loss_curve.png")
    plot_loss_curve(train_losses, val_losses, save_path=plot_save_path)

    # --- 6. Evaluate Best Model (on 16% Validation Set) ---
    if best_state is not None and training_stable:
        print(f"   -> Loading best model (from Epoch {best_epoch_num_log}) for final metrics.")
        model.load_state_dict(best_state)
        model.eval()

        # Save best model
        model_save_path = os.path.join(exp_save_dir, "best_model.pth")
        torch.save(best_state, model_save_path)
        print(f"   -> Best model saved to: {os.path.basename(model_save_path)}")

        final_preds_list = []
        with torch.no_grad():
            for i in range(0, X_val_t.size(0), batch_size):
                final_preds_list.append(model(X_val_t[i:i + batch_size].to(DEVICE)).cpu().numpy())

        y_pred_scaled = np.vstack(final_preds_list);
        y_pred_scaled = np.nan_to_num(y_pred_scaled)

        # Inverse transform to original scale
        y_true_orig = y_val_abl_raw
        y_pred_orig = y_scaler.inverse_transform(y_pred_scaled)

        print("\n   --- [ABLATION METRICS] (on 16% internal validation set) ---")
        df_metrics = evaluate_metrics(y_true_orig, y_pred_orig, print_metrics=True)

        # Save scatter plots and metrics
        scatter_plots(y_true_orig, y_pred_orig, save_dir=exp_save_dir)
        metrics_save_path = os.path.join(exp_save_dir, "metrics.csv")
        save_metrics_df(df_metrics, save_path=metrics_save_path)

        # Collect metrics for summary table
        avg_r2 = df_metrics["R2"].mean()
        avg_rpd = df_metrics["RPD"].mean()
        avg_rmse = df_metrics["RMSE"].mean()
        failed_run = False
    else:
        print(f"   -> Config FAILED or produced no valid model (Best Val Loss: {best_val_loss}).")
        avg_r2, avg_rpd, avg_rmse = [float('nan')] * 3
        best_val_loss = float('nan')
        best_epoch_num_log = 0
        failed_run = True

    print(f"   -> Config {exp_name} finished. Time to Best: {time_to_best_model:.1f}s. Avg RPD: {avg_rpd:.4f}")

    # --- 7. Collect Results (for summary table) ---
    result = {
        "Experiment": exp_name,
        "Avg_RPD": avg_rpd,
        "Avg_R2": avg_r2,
        "Avg_RMSE": avg_rmse,
        "Best_Val_Loss": best_val_loss,
        "Time_to_Best(s)": time_to_best_model,
        "Stopped_Epoch": best_epoch_num_log,
        "Failed": failed_run,
        # Attach all parameters for sorting
        **config
    }

    # Clean up memory
    del model, optimizer, criterion, scheduler
    torch.cuda.empty_cache()

    return result


# =================================================================
# --- 6. Main Execution Flow ---
# =================================================================
if __name__ == "__main__":

    all_results = []
    data_cache = {}  # Cache for preloaded and preprocessed data
    total_start_time = time.time()

    try:
        for exp_name, overrides in ABLATON_EXPERIMENTS:
            # 1. Create the final configuration for this experiment
            current_config = CHAMPION_PARAMS.copy()
            current_config.update(overrides)

            # --- Create independent save directory for this experiment ---
            safe_exp_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', exp_name)
            exp_save_dir = os.path.join(RESULT_DIR, safe_exp_name)
            os.makedirs(exp_save_dir, exist_ok=True)

            # 2. Load (or get from cache) the required data
            data_package = load_and_split_ablation_data(current_config, data_cache)

            # 3. Run training and evaluation (passing the save path)
            result_dict = train_single_experiment(exp_name, current_config, data_package, exp_save_dir)
            all_results.append(result_dict)

    except KeyboardInterrupt:
        print("\n[INFO] Ablation search interrupted by user.")
    except Exception as e:
        print(f"\n[FATAL ERROR] Main loop failed: {e}")
        import traceback;

        traceback.print_exc()

    # --- 4. Summary Report ---
    print(f"\n{'=' * 80}")
    print(f"--- ABLATION SUMMARY (V3) ---")
    print(f"{'=' * 80}")

    total_time = time.time() - total_start_time
    print(f"Total Time: {total_time:.1f}s. Ran {len(all_results)}/{len(ABLATON_EXPERIMENTS)} configurations.")

    if not all_results:
        print("[ERROR] No results were collected.")
        exit()

    # --- Create DataFrame ---
    results_df = pd.DataFrame(all_results)

    # Reorder columns: put key metrics first
    key_metrics = ["Experiment", "Avg_RPD", "Avg_R2", "Best_Val_Loss", "Time_to_Best(s)", "Stopped_Epoch", "Failed"]
    # Get all config keys dynamically (excluding those already in key_metrics)
    config_keys = [k for k in CHAMPION_PARAMS.keys() if k not in key_metrics]

    # Final column list (must ensure all columns exist)
    all_cols = key_metrics + config_keys
    final_cols = [col for col in all_cols if col in results_df.columns]

    results_df = results_df[final_cols]
    results_df.sort_values(by="Avg_RPD", ascending=False, inplace=True)

    print("\n--- [!!!] Ablation Results Summary ---")
    # Print a more readable version (showing only key metrics)
    print(results_df[key_metrics].round(4).to_string(index=False))

    # --- Save Summary Report (in V3 root directory) ---
    summary_save_path = os.path.join(RESULT_DIR, "ablation_summary.csv")
    try:
        results_df.to_csv(summary_save_path, index=False, float_format='%.6f')
        print(f"\n✅ [!!!] Final summary report saved to: {summary_save_path}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save final summary report: {e}")

    print("=" * 80)