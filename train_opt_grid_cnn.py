# train_opt_grid_cnn.py (for draft_4)
# Purpose: [!!!] Dedicated script for the Simple1DCNN model !!!
#          Executes a grid search to find the best 1D-CNN model/regularization parameters.
#          (We assume preprocessing parameters are fixed to optimal values found via SpectralMLP's ablation.)

import itertools
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split

# [!!!] Confirm correct module imports
from baselines import Simple1DCNN
from data_utils import get_data_splits, augment_data
from utils import seed_everything, evaluate_metrics, save_metrics_df
from sklearn.preprocessing import StandardScaler

# ------------- 1. Configuration (Relative Paths) -------------
# Replace absolute paths with relative paths
CSV_PATH = "cleaned_soil_data.csv"
RESULT_ROOT = "results"

RUN_NAME = "train_opt_grid_cnn"  # New explicit run name
RESULT_DIR = os.path.join(RESULT_ROOT, RUN_NAME)
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(42)

# ------------- 2. Global Search Parameters (For 1D-CNN) -------------

SEARCH_SPACE = {
    # Outer loop: Preprocessing parameters (Fixed to the optimal value)
    'sg_windows': [15],
    'sg_polyorders': [3],
    'sg_derivs': [1],

    # Inner loop: 1D-CNN Model and Training parameters
    'num_filters': [16, 32],  # Number of convolutional filters
    'filter_sizes': [7, 11, 15],  # Size of convolutional kernel
    'batch_sizes': [128],
    'dropout_rates': [0.1, 0.2],  # Dropout is typically beneficial for CNNs
    'weight_decays': [0, 1e-5]
}

# Training Control Parameters
NUM_EPOCHS = 250  # Early stopping manages the necessary length
LR = 3e-4
LOSS_FUNCTION = nn.L1Loss
EARLY_STOPPING_PATIENCE = 20
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.2
USE_AUGMENTATION = False  # Based on ablation study findings
TEST_SET_RATIO = 0.2
GRID_VAL_RATIO = 0.2  # Validation split ratio from the 80% development set

print(f"--- Running Grid Search for [Simple1DCNN] (draft_4) ---")
print(f"--- [!!!] Using Preprocessing: snv_sg_smooth (15, 3, 1) ---")
print(f"Results will be saved to: {RESULT_DIR}")
print(f"Using device: {DEVICE}")

# ------------- 3. Nested Grid Search Logic -------------
all_results = []
total_start_time = time.time()

# 3a. Outer Loop (Preprocessing)
preprocess_space = list(itertools.product(
    SEARCH_SPACE['sg_windows'],
    SEARCH_SPACE['sg_polyorders'],
    SEARCH_SPACE['sg_derivs']
))

# 3b. Inner Loop (Model)
model_space = list(itertools.product(
    SEARCH_SPACE['num_filters'],
    SEARCH_SPACE['filter_sizes'],
    SEARCH_SPACE['batch_sizes'],
    SEARCH_SPACE['dropout_rates'],
    SEARCH_SPACE['weight_decays']
))

total_preprocess_configs = len(preprocess_space)
total_model_configs = len(model_space)
total_configs = total_preprocess_configs * total_model_configs

print(
    f"Starting grid search over {total_preprocess_configs} (Preprocess) x {total_model_configs} (Model) = {total_configs} total configurations.")

config_counter = 0
data_cache = {}  # Data caching for efficiency

try:
    for (sg_w, sg_p, sg_d) in preprocess_space:

        preprocess_name = f"SG(W={sg_w}, P={sg_p}, D={sg_d})"
        print(f"\n=======================================================")
        print(f"--- LOADING DATA for Preprocessing: {preprocess_name} ---")
        print(f"=======================================================")

        preprocess_key = f"{preprocess_name}"

        # --- Load and Prepare Data (With Cache) ---
        if preprocess_key in data_cache:
            print("Loading data from cache...")
            (X_train_t, y_train_t, X_val_t, y_val_t,
             y_val_grid_raw, y_scaler, input_dim, output_dim) = data_cache[preprocess_key]
        else:
            try:
                # [!!!] Key step: Load 80% Train/Val data (20% Test is dropped here)
                X_train_val_raw, y_train_val_raw, _, _ = get_data_splits(
                    CSV_PATH,
                    preprocess_method='snv_sg_smooth',
                    test_size=TEST_SET_RATIO,
                    sg_window=sg_w,
                    sg_polyorder=sg_p,
                    sg_deriv=sg_d
                )

                # Internal split of the 80% set into Training / Validation
                X_train_grid_raw, X_val_grid_raw, y_train_grid_raw, y_val_grid_raw = train_test_split(
                    X_train_val_raw, y_train_val_raw, test_size=GRID_VAL_RATIO, random_state=42
                )

                # Data Augmentation check
                if USE_AUGMENTATION:
                    X_train_grid_aug, y_train_grid_aug = augment_data(X_train_grid_raw, y_train_grid_raw,
                                                                      random_state=42)
                else:
                    X_train_grid_aug, y_train_grid_aug = X_train_grid_raw, y_train_grid_raw

                # Fit Scalers
                X_scaler = StandardScaler().fit(X_train_grid_aug)
                y_scaler = StandardScaler().fit(y_train_grid_aug)

                # Convert to PyTorch Tensors (Scaled)
                X_train_t = torch.tensor(X_scaler.transform(X_train_grid_aug), dtype=torch.float32)
                y_train_t = torch.tensor(y_scaler.transform(y_train_grid_aug), dtype=torch.float32)
                X_val_t = torch.tensor(X_scaler.transform(X_val_grid_raw), dtype=torch.float32)
                y_val_t = torch.tensor(y_scaler.transform(y_val_grid_raw), dtype=torch.float32)

                input_dim = X_train_t.shape[1]
                output_dim = y_train_t.shape[1]

                print(f"   Data Loaded. Train Tensors: {X_train_t.shape}, Val Tensors: {X_val_t.shape}")

                # Save data to cache
                data_cache[preprocess_key] = (X_train_t, y_train_t, X_val_t, y_val_t,
                                              y_val_grid_raw, y_scaler, input_dim, output_dim)

            except Exception as e:
                print(f"[ERROR] Data loading failed for {preprocess_name}: {e}. Skipping this loop.")
                continue  # Skip this preprocessing config

        # --- [!!!] 3c. Inner Loop (Model Training) ---
        for (n_filters, f_size, bs, dr, wd) in model_space:

            config_counter += 1
            config_name = f"{preprocess_name}_Filt{n_filters}_K{f_size}_BS{bs}_DR{dr}_WD{wd}"
            print(f"\n--- Running Config {config_counter}/{total_configs}: {config_name} ---")
            config_start_time = time.time()

            # [!!!] MODIFIED: Instantiate Simple1DCNN
            model = Simple1DCNN(
                input_dim=input_dim,
                output_dim=output_dim,
                num_filters=n_filters,
                filter_size=f_size,
                dropout=dr
            ).to(DEVICE)

            optimizer = AdamW(model.parameters(), lr=LR, weight_decay=wd)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)
            criterion = LOSS_FUNCTION()

            # Initialize Early Stopping variables
            best_val_loss = float('inf')
            best_state = None
            epochs_no_improve = 0
            training_stable = True
            stopped_early = False
            best_epoch_num_log = 0

            try:
                for epoch in range(1, NUM_EPOCHS + 1):
                    # --- 1. Training Phase ---
                    model.train()
                    perm = torch.randperm(X_train_t.size(0))
                    epoch_loss = 0.0
                    cnt = 0
                    train_nan_detected = False

                    for i in range(0, len(perm), bs):
                        idx = perm[i:i + bs]
                        xb = X_train_t[idx].to(DEVICE)
                        yb = y_train_t[idx].to(DEVICE)
                        optimizer.zero_grad()
                        pred = model(xb)
                        loss = criterion(pred, yb)

                        if torch.isnan(loss) or torch.isinf(loss):
                            train_nan_detected = True;
                            continue

                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item() * xb.size(0)
                        cnt += xb.size(0)

                    if cnt == 0 and train_nan_detected:
                        print(f"  [ERROR] Epoch {epoch}: All train batches failed (NaN). Aborting config.")
                        training_stable = False;
                        break

                    train_loss = epoch_loss / cnt if cnt > 0 else float('nan')
                    if train_nan_detected or np.isnan(train_loss):
                        print(f"  [WARN] Epoch {epoch}: NaN detected in training. Aborting config.")
                        training_stable = False;
                        break

                    # --- 2. Validation Phase ---
                    model.eval()
                    val_running_loss = 0.0
                    val_cnt = 0
                    with torch.no_grad():
                        for i_val in range(0, X_val_t.size(0), bs):
                            xb_val = X_val_t[i_val:i_val + bs].to(DEVICE)
                            yb_val = y_val_t[i_val:i_val + bs].to(DEVICE)
                            out_val = model(xb_val)
                            loss_val = criterion(out_val, yb_val)

                            if not (torch.isnan(loss_val) or torch.isinf(loss_val)):
                                val_running_loss += loss_val.item() * xb_val.size(0)
                                val_cnt += xb_val.size(0)

                    val_loss = val_running_loss / val_cnt if val_cnt > 0 else float('nan')

                    # --- 3. Early Stopping and Scheduler Update ---
                    if not np.isnan(val_loss):
                        scheduler.step(val_loss)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_state = model.state_dict()  # Save best model state
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
                            print(f"   -> Stopping due to persistent NaN Val Loss.")
                            training_stable = False;
                            stopped_early = True;
                            break

                    # Log progress
                    if epoch % 40 == 0 or epoch == 1 or epoch == NUM_EPOCHS or stopped_early:
                        print(
                            f"   Epoch [{epoch}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} | VAL LOSS: {val_loss:.4f}")

            except Exception as e:
                print(f"  [ERROR] Training loop failed for {config_name}: {e}")
                training_stable = False

            # --- 3d. Validation and Result Logging ---

            if best_state is not None and training_stable:
                print(
                    f"   -> Loading best model (from Epoch {best_epoch_num_log}, Val Loss: {best_val_loss:.4f}) for final eval.")
                model.load_state_dict(best_state)
                model.eval()

                final_preds = []
                with torch.no_grad():
                    for i in range(0, X_val_t.size(0), bs):
                        final_preds.append(model(X_val_t[i:i + bs].to(DEVICE)).cpu().numpy())

                y_pred = np.vstack(final_preds)
                y_pred = np.nan_to_num(y_pred)

                y_true_orig = y_val_grid_raw
                y_pred_orig = y_scaler.inverse_transform(y_pred)

                df_metrics = evaluate_metrics(y_true_orig, y_pred_orig, print_metrics=False)
                avg_r2 = df_metrics["R2"].mean()
                avg_rpd = df_metrics["RPD"].mean()
                avg_rmse = df_metrics["RMSE"].mean()
                failed_run = False
            else:
                print(f"   -> Config failed or produced no valid model (Best Val Loss: {best_val_loss}).")
                avg_r2, avg_rmse, avg_rpd = [float('nan')] * 3
                failed_run = True

            config_time = time.time() - config_start_time

            # Collect 1D-CNN parameters
            all_results.append([
                # Preprocessing
                sg_w, sg_p, sg_d,
                # Model
                n_filters, f_size,
                # Training
                bs, dr, wd,
                # Metrics
                avg_rpd, avg_r2, avg_rmse, config_time, failed_run
            ])

            print(f"   -> Config {config_counter} finished. Time: {config_time:.1f}s. Avg RPD: {avg_rpd:.4f}")
            del model, optimizer, criterion, scheduler
            torch.cuda.empty_cache()

        # Clean up data cache (if multiple preprocessing combos were searched)
        if len(preprocess_space) > 1:
            # Note: This logic assumes data_cache stores tensors, so cleanup is necessary if multi-loop
            data_cache.pop(preprocess_key)
            torch.cuda.empty_cache()

except KeyboardInterrupt:
    print("\n[INFO] Grid search interrupted by user.")

# ------------- 4. Final Summary Report -------------
print(f"\n=======================================================")
print(f"--- 1D-CNN GRID SEARCH SUMMARY ---")
print(f"=======================================================")
total_time = time.time() - total_start_time
print(f"Total Time: {total_time:.1f}s. Ran {config_counter}/{total_configs} configurations.")

if not all_results:
    print("[ERROR] No results were collected.")
    exit()

# Update DataFrame columns
results_df = pd.DataFrame(all_results,
                          columns=["SG_W", "SG_P", "SG_D",
                                   "NumFilters", "FilterSize", "BatchSize", "Dropout", "WeightDecay",
                                   "RPD_mean", "R2_mean", "RMSE_mean", "Time_s", "Failed"])

results_df.sort_values(by="RPD_mean", ascending=False, inplace=True)

print("\n--- Grid Search Results (Top 20) ---")
print(results_df.head(20).round(4).to_string(index=False))

# Save summary report
summary_save_path = os.path.join(RESULT_DIR, "cnn_grid_search_summary.csv")
try:
    results_df.to_csv(summary_save_path, index=False, float_format='%.6f')
    print(f"\n✅ Final summary report saved to: {summary_save_path}")
except Exception as e:
    print(f"\n[ERROR] Failed to save final summary report: {e}")
print("=======================================================")