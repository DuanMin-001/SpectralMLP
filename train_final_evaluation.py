# train_final_evaluation.py (for draft_4)
# Purpose: Executes the final, unique evaluation of the *Champion Model* SpectralMLP on the 20% independent test set.

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

# Import core dependencies
from data_utils import get_data_splits, augment_data
from model import SpectralMLP
from utils import seed_everything, plot_loss_curve, scatter_plots, evaluate_metrics, save_metrics_df

# ------------- Configuration (Relative Paths and Setup) -------------
# Replace absolute paths with relative paths
CSV_PATH = "cleaned_soil_data.csv"
RESULT_ROOT = "results"
RUN_NAME = "train_final_evaluation"  # Explicit run name
RESULT_DIR = os.path.join(RESULT_ROOT, RUN_NAME)
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Note: Seed 44 used for stability test runs
seed_everything(42)

# Champion Model Parameters (from ablation study)
EMBED_DIM = 128
DEPTH = 2
DROPOUT_RATE = 0.0
WEIGHT_DECAY = 0.0
PREPROCESS = 'snv_sg_smooth'
SG_WINDOW = 15
SG_POLYORDER = 3
SG_DERIV = 1
USE_AUGMENTATION = False # Critical finding from ablation

# Training Control Parameters
EPOCHS = 250
BATCH_SIZE = 128
LR = 3e-4
LOSS_FUNCTION = nn.L1Loss
EARLY_STOPPING_PATIENCE = 20
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.2
TEST_SET_RATIO = 0.2
FINAL_VAL_RATIO = 0.15 # Validation split ratio from the 80% development set

print(f"--- Running FINAL EVALUATION on [SpectralMLP] (draft_4) ---")
print(f"--- Baseline: Data Augmentation = {USE_AUGMENTATION} ---")
print(f"Model: SpectralMLP (embed={EMBED_DIM}, depth={DEPTH}, dropout={DROPOUT_RATE})")
print(f"Preprocessing: {PREPROCESS} (SG W={SG_WINDOW}, P={SG_POLYORDER}, D={SG_DERIV})")
print(f"Results will be saved to: {RESULT_DIR}")
print(f"Using device: {DEVICE}")

# ------------- 1. Data Loading (Load 80% Train/Val and 20% Test Set) -------------
try:
    print("Loading all data splits...")
    # X_train_val_raw: 80% development set; X_test_raw: 20% independent test set
    X_train_val_raw, y_train_val_raw, X_test_raw, y_test_raw = get_data_splits(
        CSV_PATH, preprocess_method=PREPROCESS, test_size=TEST_SET_RATIO,
        sg_window=SG_WINDOW, sg_polyorder=SG_POLYORDER, sg_deriv=SG_DERIV
    )

    print(f"   Initial Train/Val set (80%): {X_train_val_raw.shape}")
    print(f"   Initial Test set (20%): {X_test_raw.shape}")

    # ------------- 2. Data Resplitting and Standardization (Prevent Data Leakage) -------------

    print(f"Splitting 80% set into final train/val (Ratio={FINAL_VAL_RATIO})...")
    # This split creates the final training set (~68%) and the final validation set (~12% for early stopping)
    X_train_final_raw, X_val_final_raw, y_train_final_raw, y_val_final_raw = train_test_split(
        X_train_val_raw, y_train_val_raw, test_size=FINAL_VAL_RATIO, random_state=42
    )

    if USE_AUGMENTATION:
        X_train_aug, y_train_aug = augment_data(X_train_final_raw, y_train_final_raw, random_state=42)
    else:
        X_train_aug, y_train_aug = X_train_final_raw, y_train_final_raw

    # Fit Scalers ONLY on the final training set (X_train_aug)
    print("Fitting scalers ONLY on final train set...")
    X_scaler = StandardScaler().fit(X_train_aug)
    y_scaler = StandardScaler().fit(y_train_aug)

    print("Transforming all data sets...")
    # Scale all splits using the fitted scalers
    X_train_t = torch.tensor(X_scaler.transform(X_train_aug), dtype=torch.float32)
    y_train_t = torch.tensor(y_scaler.transform(y_train_aug), dtype=torch.float32)
    X_val_t = torch.tensor(X_scaler.transform(X_val_final_raw), dtype=torch.float32)
    y_val_t = torch.tensor(y_scaler.transform(y_val_final_raw), dtype=torch.float32)
    X_test_t = torch.tensor(X_scaler.transform(X_test_raw), dtype=torch.float32)

    print(f"   Final Train Tensors: {X_train_t.shape}")
    print(f"   Final Val Tensors (for stopping): {X_val_t.shape}")
    print(f"   Final Test Tensors (for reporting): {X_test_t.shape}")

except Exception as e:
    print(f"[ERROR] Failed to load or preprocess data: {e}"); exit()

input_dim = X_train_t.shape[1]
output_dim = y_train_t.shape[1]

# ------------- 3. Model and Training Configuration -------------
# Instantiate SpectralMLP (Champion Model)
model = SpectralMLP(input_dim=input_dim, embed_dim=EMBED_DIM, depth=DEPTH, output_dim=output_dim,
                    dropout=DROPOUT_RATE).to(DEVICE)
criterion = LOSS_FUNCTION()
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)

# ------------- 4. Training (on ~68%, Early Stopping on ~12%) -------------
train_losses, val_losses = [], []
best_val_loss = float('inf'); best_state = None
epochs_no_improve = 0; start_time = time.time()
training_stable = True; best_epoch_num_log = 0; stopped_early = False

print("--- Starting FINAL Training ---")
print("   (Training on final train set, Validating on final val set for early stopping)")

try:
    for epoch in range(1, EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        perm = torch.randperm(X_train_t.size(0))
        running_loss = 0.0; cnt = 0; epoch_stable = True
        for i in range(0, len(perm), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = X_train_t[idx].to(DEVICE); yb = y_train_t[idx].to(DEVICE)
            optimizer.zero_grad(); out = model(xb); loss = criterion(out, yb)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [WARN] Epoch {epoch}: NaN train loss. Skipping batch.")
                epoch_stable = False; continue
            loss.backward(); optimizer.step()
            running_loss += loss.item() * xb.size(0); cnt += xb.size(0)

        if cnt == 0 and not epoch_stable:
            train_loss = float('nan'); training_stable = False
            print(f"[ERROR] Epoch {epoch}: All train batches failed (NaN). Aborting.")
            break
        else:
            train_loss = running_loss / cnt if cnt > 0 else float('nan')
        train_losses.append(train_loss)

        # --- Evaluation Phase (on X_val_t for stopping) ---
        model.eval()
        val_running_loss = 0.0; val_cnt = 0
        with torch.no_grad():
            for i_val in range(0, X_val_t.size(0), BATCH_SIZE):
                xb_val = X_val_t[i_val:i_val + BATCH_SIZE].to(DEVICE)
                yb_val = y_val_t[i_val:i_val + BATCH_SIZE].to(DEVICE)
                out_val_batch = model(xb_val); loss_val_batch = criterion(out_val_batch, yb_val)
                if torch.isnan(loss_val_batch) or torch.isinf(loss_val_batch):
                    # Penalize severely if loss is NaN/Inf
                    val_running_loss += 1e9 * xb_val.size(0)
                else:
                    val_running_loss += loss_val_batch.item() * xb_val.size(0)
                val_cnt += xb_val.size(0)
        val_loss = val_running_loss / val_cnt if val_cnt > 0 else float('nan')
        val_losses.append(val_loss)

        # --- Early Stopping and Model Saving (Based on Validation Set) ---
        if not np.isnan(val_loss):
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss; best_state = model.state_dict()
                torch.save(best_state, os.path.join(RESULT_DIR, "best_final_model.pth"))
                epochs_no_improve = 0; best_epoch_num_log = epoch
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print(f"\n[INFO] Early stopping at epoch {epoch} (based on validation set).")
                    stopped_early = True; break
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE * 2:
                print(f"\n[WARN] Stopping due to persistent NaN/Inf val loss at epoch {epoch}.")
                training_stable = False; stopped_early = True; break

        current_lr = optimizer.param_groups[0]['lr']
        if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS or stopped_early:
            print(f"Epoch [{epoch}/{EPOCHS}] Train Loss: {train_loss:.4f} | VAL LOSS: {val_loss:.4f} | LR: {current_lr:.6f}")
        if not training_stable or (np.isnan(train_loss) and epoch > 5):
            print(f"[ERROR] Training unstable. Aborting."); training_stable = False; break

except KeyboardInterrupt:
    print("\n[INFO] Training interrupted by user.")
except Exception as e:
    print(f"[ERROR] Training loop failed critically: {e}");
    import traceback; traceback.print_exc(); training_stable = False

# ------------- 5. Final Evaluation -------------
print("\n--- Final Evaluation Finished ---")
elapsed = time.time() - start_time
print(f"Total Training Time: {elapsed:.1f}s")
if stopped_early: print(f"Stopped early at epoch: {best_epoch_num_log}")

plot_loss_curve(train_losses, val_losses, save_path=os.path.join(RESULT_DIR, "loss_curve_final_val.png"))

if best_state is not None and training_stable:
    print(f"Loading best model (from Epoch {best_epoch_num_log}) with Val Loss: {best_val_loss:.4f}")
    model.load_state_dict(best_state)
elif training_stable:
    print("[WARN] No best model was saved. Using final model state.")
    best_epoch_num_log = EPOCHS if not stopped_early else epoch
else:
    print("[ERROR] Training was unstable. Evaluation skipped."); exit()

# [!!!] Execute Final Prediction on 20% Test Set (X_test_t)
model.eval()
final_preds_list = []
with torch.no_grad():
    for i in range(0, X_test_t.size(0), BATCH_SIZE):
        final_preds_list.append(model(X_test_t[i:i + BATCH_SIZE].to(DEVICE)).cpu().numpy())
if not final_preds_list: print("[ERROR] No predictions were generated."); exit()

final_y_pred_scaled = np.vstack(final_preds_list); final_y_pred_scaled = np.nan_to_num(final_y_pred_scaled)

# Inverse transform to original scale
try:
    final_y_pred_orig = y_scaler.inverse_transform(final_y_pred_scaled)
    final_y_true_orig = y_test_raw
except Exception as e:
    print(f"[ERROR] Inverse transform failed: {e}"); exit()

# --- Report Final Metrics (on Independent Test Set) ---
print(f"\n--- [FINAL METRICS] Independent Test Set (Best Model from Epoch {best_epoch_num_log}) ---")
scatter_plots(final_y_true_orig, final_y_pred_orig, save_dir=RESULT_DIR)
final_metrics_df = evaluate_metrics(final_y_true_orig, final_y_pred_orig, print_metrics=True)
save_metrics_df(final_metrics_df, os.path.join(RESULT_DIR, "metrics_summary_FINAL_TEST.csv"))

final_avg_r2 = final_metrics_df['R2'].mean()
final_avg_rpd = final_metrics_df['RPD'].mean()

print("\n=======================================================")
print(f"✅ FINAL EVALUATION COMPLETE (draft_4 - SpectralMLP)")
print(f"   Best Model Avg R2 (on Test Set): {final_avg_r2:.4f}")
print(f"   Best Model Avg RPD (on Test Set): {final_avg_rpd:.4f}")
print(f"   Total Time: {elapsed:.1f}s")
print(f"   All final results saved to: {RESULT_DIR}")
print("=======================================================")
if not training_stable: print("[WARN] Training encountered instability.")