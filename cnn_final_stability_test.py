# cnn_final_stability_test_automated.py (Optimized for Automated Stability Analysis)
# Purpose: Executes final evaluation of Simple1DCNN on the 20% independent test set,
#          automatically iterating through seeds [42, 43, 44] to generate stability metrics.

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
import itertools

# --- Import Custom Modules ---
from data_utils import get_data_splits, augment_data
from baselines import Simple1DCNN
from utils import seed_everything, plot_loss_curve, scatter_plots, evaluate_metrics, save_metrics_df

# ------------- Configuration (Relative Paths) -------------
CSV_PATH = "cleaned_soil_data.csv"
RESULT_ROOT = "results"
RUN_NAME = "cnn_final_stability"
BASE_RESULT_DIR = os.path.join(RESULT_ROOT, RUN_NAME)  # Base directory for ALL stability runs
os.makedirs(BASE_RESULT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Stability Test Setup ---
STABILITY_SEEDS = [42, 43, 44]  # Seeds for stability test (Table 4)

# --- Model Parameters (Fixed Optimal) ---
PREPROCESS = 'snv_sg_smooth'
SG_WINDOW = 15
SG_POLYORDER = 3
SG_DERIV = 1
USE_AUGMENTATION = False

# 1D-CNN Champion Parameters (from grid search)
CNN_WEIGHT_DECAY = 1e-5
CNN_PARAMS = {
    'num_filters': 32,
    'filter_size': 15,
    'dropout': 0.1,
}

# Training Control Parameters (Fixed)
EPOCHS = 250
BATCH_SIZE = 128
LR = 3e-4
LOSS_FUNCTION = nn.L1Loss
EARLY_STOPPING_PATIENCE = 20
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.2
TEST_SET_RATIO = 0.2
FINAL_VAL_RATIO = 0.15  # 15% split from the 80% set for validation

print(f"--- Running AUTOMATED Stability Test (Seeds: {STABILITY_SEEDS}) ---")
print(f"Model: Simple1DCNN | Base Results Directory: {BASE_RESULT_DIR}")

# --- Global Data Preparation (Must be done once) ---
try:
    X_train_val_raw, y_train_val_raw, X_test_raw, y_test_raw = get_data_splits(
        CSV_PATH, preprocess_method=PREPROCESS, test_size=TEST_SET_RATIO,
        random_state=42,  # FIXED SEED for the data split
        sg_window=SG_WINDOW, sg_polyorder=SG_POLYORDER, sg_deriv=SG_DERIV
    )
except Exception as e:
    print(f"[FATAL ERROR] Failed to load data: {e}");
    exit()

input_dim = X_train_val_raw.shape[1]
output_dim = y_train_val_raw.shape[1]

# --- Storage for Final Summary ---
stability_results = []
all_runs_metrics = []
total_start_time = time.time()

# =================================================================
# --- 1. Automated Stability Loop ---
# =================================================================
for seed in STABILITY_SEEDS:
    current_run_name = f"seed_{seed}"

    # [!!!] KEY CHANGE: Define individual result directory for the current seed
    CURRENT_RUN_DIR = os.path.join(BASE_RESULT_DIR, current_run_name)
    os.makedirs(CURRENT_RUN_DIR, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"--- Starting Stability Run: {current_run_name} (Dir: {os.path.basename(CURRENT_RUN_DIR)}) ---")
    print(f"{'=' * 70}")

    seed_everything(seed)
    run_start_time = time.time()

    # --- 1a. Data Resplitting and Standardization ---
    X_train_final_raw, X_val_final_raw, y_train_final_raw, y_val_final_raw = train_test_split(
        X_train_val_raw, y_train_val_raw, test_size=FINAL_VAL_RATIO, random_state=seed
        # Use current seed for internal split/shuffle
    )

    X_scaler = StandardScaler().fit(X_train_final_raw)
    y_scaler = StandardScaler().fit(y_train_final_raw)

    # Transform all data sets
    X_train_t = torch.tensor(X_scaler.transform(X_train_final_raw), dtype=torch.float32)
    y_train_t = torch.tensor(y_scaler.transform(y_train_final_raw), dtype=torch.float32)
    X_val_t = torch.tensor(X_scaler.transform(X_val_final_raw), dtype=torch.float32)
    y_val_t = torch.tensor(y_scaler.transform(y_val_final_raw), dtype=torch.float32)
    X_test_t = torch.tensor(X_scaler.transform(X_test_raw), dtype=torch.float32)  # Test data transformation

    # --- 1b. Model Configuration and Training Setup ---
    model = Simple1DCNN(input_dim=input_dim, output_dim=output_dim, **CNN_PARAMS).to(DEVICE)
    criterion = LOSS_FUNCTION()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=CNN_WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)

    # Training State Tracking
    best_val_loss = float('inf');
    best_state = None
    epochs_no_improve = 0;
    best_epoch_num_log = 0;
    training_stable = True
    train_losses, val_losses = [], []

    # --- 1c. Training Loop ---
    try:
        for epoch in range(1, EPOCHS + 1):
            # --- Training Phase ---
            model.train()
            perm = torch.randperm(X_train_t.size(0))
            running_loss = 0.0;
            cnt = 0;
            epoch_stable = True
            for i in range(0, len(perm), BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE];
                xb = X_train_t[idx].to(DEVICE);
                yb = y_train_t[idx].to(DEVICE)
                optimizer.zero_grad();
                out = model(xb);
                loss = criterion(out, yb)
                if torch.isnan(loss) or torch.isinf(loss):
                    epoch_stable = False;
                    continue
                loss.backward();
                optimizer.step()
                running_loss += loss.item() * xb.size(0);
                cnt += xb.size(0)

            if cnt == 0 and not epoch_stable: training_stable = False; break
            train_loss = running_loss / cnt if cnt > 0 else float('nan');
            train_losses.append(train_loss)
            if np.isnan(train_loss): training_stable = False; break

            # --- Evaluation Phase (Validation Set) ---
            model.eval();
            val_running_loss = 0.0;
            val_cnt = 0
            with torch.no_grad():
                for i_val in range(0, X_val_t.size(0), BATCH_SIZE):
                    xb_val = X_val_t[i_val:i_val + BATCH_SIZE].to(DEVICE);
                    yb_val = y_val_t[i_val:i_val + BATCH_SIZE].to(DEVICE)
                    out_val_batch = model(xb_val);
                    loss_val_batch = criterion(out_val_batch, yb_val)
                    if not (torch.isnan(loss_val_batch) or torch.isinf(loss_val_batch)):
                        val_running_loss += loss_val_batch.item() * xb_val.size(0);
                        val_cnt += xb_val.size(0)

            val_loss = val_running_loss / val_cnt if val_cnt > 0 else float('nan');
            val_losses.append(val_loss)

            # --- Early Stopping and Model Saving ---
            if not np.isnan(val_loss):
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss;
                    best_state = model.state_dict()
                    # Save model to the seed-specific directory
                    torch.save(best_state, os.path.join(CURRENT_RUN_DIR, "best_model.pth"))
                    epochs_no_improve = 0;
                    best_epoch_num_log = epoch
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= EARLY_STOPPING_PATIENCE: break
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE * 2: training_stable = False; break

            if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
                print(f"Epoch [{epoch}/{EPOCHS}] Train Loss: {train_loss:.4f} | VAL LOSS: {val_loss:.4f}")
            if not training_stable or (np.isnan(train_loss) and epoch > 5): break

    except Exception as e:
        print(f"[ERROR] Training loop failed critically for seed {seed}: {e}");
        training_stable = False
        import traceback;

        traceback.print_exc()

    # --- 1d. Final Evaluation on Test Set ---
    elapsed_time = time.time() - run_start_time

    if best_state is not None and training_stable:
        model.load_state_dict(best_state)
    elif training_stable:
        print(f"[WARN] No best model saved for seed {seed}. Using final model state.")
    else:
        print(f"[ERROR] Run for seed {seed} unstable. Skipping evaluation.")
        # Store NaN results for structural completeness
        final_metrics_df = pd.DataFrame([{"R2": np.nan, "RPD": np.nan, "MAE": np.nan}])
        stability_results.append(
            {"Seed": seed, "R2_mean": np.nan, "RPD_mean": np.nan, "Time_s": elapsed_time, "Epochs": 0})
        continue

    # Execute Prediction
    model.eval()
    final_preds_list = []
    with torch.no_grad():
        for i in range(0, X_test_t.size(0), BATCH_SIZE):
            final_preds_list.append(model(X_test_t[i:i + BATCH_SIZE].to(DEVICE)).cpu().numpy())

    final_y_pred_scaled = np.vstack(final_preds_list);
    final_y_pred_scaled = np.nan_to_num(final_y_pred_scaled)
    final_y_pred_orig = y_scaler.inverse_transform(final_y_pred_scaled)
    final_y_true_orig = y_test_raw

    # Report Metrics
    final_metrics_df = evaluate_metrics(final_y_true_orig, final_y_pred_orig, print_metrics=False)

    # Store results for final summary table (Table 4)
    stability_results.append({
        "Seed": seed,
        "R2_mean": final_metrics_df['R2'].mean(),
        "RPD_mean": final_metrics_df['RPD'].mean(),
        "Time_s": elapsed_time,
        "Epochs": best_epoch_num_log
    })

    # Save individual report (optional, but good for checks)
    metrics_save_path = os.path.join(CURRENT_RUN_DIR, "metrics_summary.csv")
    save_metrics_df(final_metrics_df, save_path=metrics_save_path)

    # Save loss curve (specific to this seed's directory)
    plot_save_path = os.path.join(CURRENT_RUN_DIR, "loss_curve.png")
    plot_loss_curve(train_losses, val_losses, save_path=plot_save_path)

    print(f"--- Run {seed} Finished. Time: {elapsed_time:.1f}s. Avg RPD: {stability_results[-1]['RPD_mean']:.4f} ---")

    # Clean up memory
    del model, optimizer, criterion, scheduler, X_train_t, y_train_t, X_val_t, y_val_t
    torch.cuda.empty_cache()

# =================================================================
# --- 2. Final Summary Calculation (Table 4 Generation) ---
# =================================================================

print(f"\n{'=' * 70}")
print(f"--- FINAL STABILITY SUMMARY (TABLE 4 DATA) ---")
print(f"{'=' * 70}")

df_results = pd.DataFrame(stability_results)

# Calculate final Mean and Std Dev for R2 and RPD
final_r2_mean = df_results['R2_mean'].mean()
final_r2_std = df_results['R2_mean'].std()
final_rpd_mean = df_results['RPD_mean'].mean()
final_rpd_std = df_results['RPD_mean'].std()

# Format final output for Table 4 inclusion (Mean (± Std))
summary_output = {
    "Model": "Simple1DCNN",
    "R2": f"{final_r2_mean:.3f} (± {final_r2_std:.3f})",
    "RPD": f"{final_rpd_mean:.3f} (± {final_rpd_std:.3f})"
}

# Print summary table and save to file (in the base results directory)
print(f"Final Mean R2: {final_r2_mean:.4f} (Std: {final_r2_std:.4f})")
print(f"Final Mean RPD: {final_rpd_mean:.4f} (Std: {final_rpd_std:.4f})")

df_results_save = os.path.join(BASE_RESULT_DIR, "stability_summary_table4.csv")
pd.DataFrame([summary_output]).to_csv(df_results_save, index=False)
df_results.to_csv(os.path.join(BASE_RESULT_DIR, "stability_raw_all_runs.csv"), index=False)

print(f"\n✅ Stability Analysis Complete. Table 4 data saved to: {df_results_save}")
print("=" * 70)