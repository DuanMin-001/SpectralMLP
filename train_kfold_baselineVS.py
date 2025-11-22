# train_kfold_baselineVS.py (for draft_4 - V5_FAIR_FIXED)
# Purpose: Executes 5-Fold Cross-Validation on the 80% Train/Val Set for *all baseline models*.
# MODIFIED (V5): Fixes the critical SVR double-scaling error and ensures fair comparison across all models.

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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --- Import Custom Modules ---
from model import SpectralMLP
from baselines import PLSBaseline, SVRBaseline, RandomForestBaseline, Simple1DCNN
from data_utils import get_data_splits, augment_data
from utils import seed_everything, evaluate_metrics, save_metrics_df

# ------------- 1. Configuration (Relative Paths) -------------
# Use relative paths
CSV_PATH = "cleaned_soil_data.csv"
RESULT_ROOT = "results"
RUN_NAME = "train_kfold_baselineVS"
RESULT_DIR = os.path.join(RESULT_ROOT, RUN_NAME)
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything(42)

N_SPLITS = 5
TARGETS = ["pH", "OC", "N"]

MODELS_TO_RUN = [
    "SpectralMLP",
    "PLSBaseline",
    "SVRBaseline",
    "RandomForestBaseline",
    "Simple1DCNN"
]

# ------------- 2. Fixed Optimal Preprocessing (from ablation study) -------------
PREPROCESS = 'snv_sg_smooth'
SG_WINDOW = 15
SG_POLYORDER = 3
SG_DERIV = 1
USE_AUGMENTATION = False
TEST_SET_RATIO = 0.2

# ------------- 3. Champion Parameters (from ablation study and K-Fold) -------------

# PyTorch Training Control
EPOCHS = 250
LOSS_FUNCTION = nn.L1Loss
EARLY_STOPPING_PATIENCE = 20
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.2

# SpectralMLP Champion Parameters
CHAMPION_PARAMS_MLP = {
    'embed_dim': 128,
    'depth': 2,
    'dropout_rate': 0.0,
    'weight_decay': 0.0,
    'batch_size': 128,
    'lr': 3e-4
}

# 1D-CNN Champion Parameters
CHAMPION_PARAMS_CNN = {
    'num_filters': 32,
    'filter_size': 15,
    'dropout_rate': 0.1,
    'weight_decay': 1e-5,
    'batch_size': 128,
    'lr': 3e-4
}

print(f"--- Running K-Fold (N={N_SPLITS}) Baseline Comparison (draft_4 - V5_FIXED) ---")
print(f"Comparing Models: {MODELS_TO_RUN}")
print(f"Fixed Preprocessing: {PREPROCESS} (SG W={SG_WINDOW}, P={SG_POLYORDER}, D={SG_DERIV})")
print(f"--- Using Champion Loss: L1Loss ---")
print(f"Results will be saved to: {RESULT_DIR}")
print(f"Using device: {DEVICE}")

# ------------- 4. Data Loading (Loads 80% Development Set) -------------
try:
    print("Loading 80% Train/Val data (and IGNORING 20% Test data)...")
    # Load the 80% development set, discarding the 20% independent test set
    X_train_val_raw, y_train_val_raw, _, _ = get_data_splits(
        CSV_PATH, preprocess_method=PREPROCESS, test_size=TEST_SET_RATIO,
        random_state=42,
        sg_window=SG_WINDOW, sg_polyorder=SG_POLYORDER, sg_deriv=SG_DERIV
    )
    print(f"   Loaded Train/Val set (80%) shape: {X_train_val_raw.shape}")
    print(f"   (20% Independent Test Set is HELD OUT)")

except Exception as e:
    print(f"[ERROR] Failed to load data: {e}")
    exit()

# ------------- 5. K-Fold CV Loop (Iterate All Models) -------------
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

all_models_metrics_dfs = []
total_cv_start_time = time.time()

# --- Iterate through Models ---
for model_name in MODELS_TO_RUN:

    seed_everything(42)

    print(f"\n=======================================================")
    print(f"--- Starting K-Fold CV for MODEL: {model_name} ---")
    print(f"=======================================================")

    model_start_time = time.time()
    all_fold_metrics = []

    # --- Iterate through K-Fold Splits ---
    for fold, (train_fold_idx, val_fold_idx) in enumerate(kf.split(X_train_val_raw, y_train_val_raw)):

        fold_start_time = time.time()
        print(f"\n--- Fold {fold + 1}/{N_SPLITS} for {model_name} ---")

        # 5a. Split Data for Current Fold
        X_train_fold_raw = X_train_val_raw[train_fold_idx]
        y_train_fold_raw = y_train_val_raw[train_fold_idx]
        X_val_fold_raw = X_train_val_raw[val_fold_idx]
        y_val_fold_raw = y_train_val_raw[val_fold_idx]

        # 5b. Standardization (Fit only on Fold-Train)
        X_scaler = StandardScaler().fit(X_train_fold_raw)
        y_scaler = StandardScaler().fit(y_train_fold_raw)

        # Scale X and Y data for PyTorch models
        X_train_scaled = X_scaler.transform(X_train_fold_raw)
        y_train_scaled = y_scaler.transform(y_train_fold_raw)
        X_val_scaled = X_scaler.transform(X_val_fold_raw)

        input_dim = X_train_scaled.shape[1]
        output_dim = y_train_scaled.shape[1]

        # 5c. Model Training/Fitting
        training_stable = True
        y_true_orig, y_pred_orig = None, None

        # ---------------
        # A. Sklearn Models (PLS, SVR, RF) - Trained on RAW (X) and RAW (Y)
        # ---------------
        if model_name in ["PLSBaseline", "SVRBaseline", "RandomForestBaseline"]:
            try:
                print(f"Fitting {model_name} with nested GridSearchCV...")
                model = None

                if model_name == "PLSBaseline":
                    # ... (PLS logic remains the same) ...
                    model = PLSBaseline()
                    model.fit(X_train_fold_raw, y_train_fold_raw)
                    y_pred_orig = model.predict(X_val_fold_raw)

                elif model_name == "SVRBaseline":
                    # [!!!] NOTE FOR PUBLIC RELEASE:
                    # The dedicated SVRBaseline class uses a reduced GridSearch space and
                    # standard StandardScaler (with_mean=True) for feasibility,
                    # as the strict with_mean=False/full GridSearch requires >18 hours per fold.
                    print("   (Using SVR with reduced GridSearch for efficiency)")

                    model = SVRBaseline()
                    model.fit(X_train_fold_raw, y_train_fold_raw)
                    y_pred_orig = model.predict(X_val_fold_raw)

                elif model_name == "RandomForestBaseline":
                    # ... (RF logic remains the same) ...
                    model = RandomForestBaseline()
                    model.fit(X_train_fold_raw, y_train_fold_raw)
                    y_pred_orig = model.predict(X_val_fold_raw)

                y_true_orig = y_val_fold_raw  # True values are unscaled for comparison

            except Exception as e:
                print(f"[ERROR] Sklearn model {model_name} fold {fold + 1} FAILED: {e}")
                import traceback

                traceback.print_exc()

        # ---------------
        # B. PyTorch Models (SpectralMLP, 1D-CNN) - Trained on SCALED (X) and SCALED (Y)
        # ---------------
        elif model_name in ["SpectralMLP", "Simple1DCNN"]:

            # Prepare Tensors and DataLoaders
            X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
            y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
            X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
            # Y_val target must be scaled for loss calculation
            y_val_t = torch.tensor(y_scaler.transform(y_val_fold_raw), dtype=torch.float32)

            # Model and Parameter Setup
            params = CHAMPION_PARAMS_MLP if model_name == "SpectralMLP" else CHAMPION_PARAMS_CNN

            if model_name == "SpectralMLP":
                model = SpectralMLP(input_dim=input_dim, output_dim=output_dim,
                                    embed_dim=params['embed_dim'], depth=params['depth'],
                                    dropout=params['dropout_rate']).to(DEVICE)
            else:  # Simple1DCNN
                model = Simple1DCNN(input_dim=input_dim, output_dim=output_dim,
                                    num_filters=params['num_filters'], filter_size=params['filter_size'],
                                    dropout=params['dropout_rate']).to(DEVICE)

            criterion = LOSS_FUNCTION()
            optimizer = AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)
            bs = params['batch_size']

            # Training Loop (PyTorch training implementation remains the same)
            best_val_loss, best_state, epochs_no_improve = float('inf'), None, 0

            try:
                for epoch in range(1, EPOCHS + 1):
                    model.train()
                    perm = torch.randperm(X_train_t.size(0))
                    running_loss, cnt = 0.0, 0

                    # Training step implementation (forward, backward, step)
                    for i in range(0, len(perm), bs):
                        idx = perm[i:i + bs];
                        xb = X_train_t[idx].to(DEVICE);
                        yb = y_train_t[idx].to(DEVICE)
                        optimizer.zero_grad();
                        out = model(xb);
                        loss = criterion(out, yb)
                        if torch.isnan(loss) or torch.isinf(loss): continue
                        loss.backward();
                        optimizer.step()
                        running_loss += loss.item() * xb.size(0);
                        cnt += xb.size(0)

                    # Validation step implementation (with torch.no_grad)
                    model.eval()
                    val_running_loss, val_cnt = 0.0, 0
                    with torch.no_grad():
                        for i_val in range(0, X_val_t.size(0), bs):
                            xb_val = X_val_t[i_val:i_val + bs].to(DEVICE);
                            yb_val = y_val_t[i_val:i_val + bs].to(DEVICE)
                            out_val = model(xb_val);
                            loss_val = criterion(out_val, yb_val)
                            if not (torch.isnan(loss_val) or torch.isinf(loss_val)):
                                val_running_loss += loss_val.item() * xb_val.size(0);
                                val_cnt += xb_val.size(0)

                    val_loss = val_running_loss / val_cnt if val_cnt > 0 else float('nan')

                    # Early Stopping & Scheduler logic
                    if not np.isnan(val_loss):
                        scheduler.step(val_loss)
                        if val_loss < best_val_loss:
                            best_val_loss, best_state, epochs_no_improve = val_loss, model.state_dict(), 0
                        else:
                            epochs_no_improve += 1
                            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                                print(f"  [INFO] Fold {fold + 1}: Early stopping at epoch {epoch}.")
                                break
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= EARLY_STOPPING_PATIENCE * 2:
                            print(f"  [WARN] Fold {fold + 1}: Stopping due to persistent NaN val loss.")
                            training_stable = False;
                            break

            except Exception as e:
                print(f"  [ERROR] Fold {fold + 1} training loop failed critically: {e}")
                training_stable = False

            # Post-training Evaluation
            if best_state is not None and training_stable:
                model.load_state_dict(best_state)
            elif training_stable:
                print(f"  [WARN] Fold {fold + 1}: No best model saved. Using final model state for metrics.")
            else:
                print(f"  [ERROR] Fold {fold + 1}: Training unstable. Skipping eval.")
                y_true_orig, y_pred_orig = None, None

            if training_stable and y_true_orig is None:  # Only proceed if training was stable and eval variables are needed
                model.eval()
                fold_preds_list = []

                # --- PyTorch CRITICAL: Use DataLoader for efficient prediction ---
                val_predict_ds = torch.utils.data.TensorDataset(X_val_t)
                val_predict_loader = torch.utils.data.DataLoader(val_predict_ds, batch_size=bs * 2, shuffle=False)

                with torch.no_grad():
                    for xb_val_tuple in val_predict_loader:
                        xb_val = xb_val_tuple[0]
                        fold_preds_list.append(model(xb_val.to(DEVICE)).cpu().numpy())

                y_pred_scaled = np.vstack(fold_preds_list)
                y_pred_scaled = np.nan_to_num(y_pred_scaled)

                # --- PyTorch CRITICAL: Inverse transform predictions ---
                y_pred_orig = y_scaler.inverse_transform(y_pred_scaled)
                y_true_orig = y_val_fold_raw  # Use original unscaled validation targets

        else:
            print(f"[ERROR] Unknown model_name: {model_name}. Skipping fold.")
            y_true_orig, y_pred_orig = None, None

        # 5d. Fold Evaluation and Cleanup
        fold_elapsed = time.time() - fold_start_time
        print(f"--- Fold {fold + 1} Finished. Time: {fold_elapsed:.1f}s ---")

        if y_true_orig is not None and y_pred_orig is not None:
            print(f"   Metrics for {model_name} Fold {fold + 1}:")
            fold_metrics_df = evaluate_metrics(y_true_orig, y_pred_orig,
                                               print_metrics=True)
            all_fold_metrics.append(fold_metrics_df)
        else:
            print(f"   [ERROR] Skipping metrics for Fold {fold + 1} due to error.")
            # Append NaN metrics to maintain DataFrame structure
            nan_metrics = pd.DataFrame([["pH", np.nan, np.nan, np.nan, np.nan],
                                        ["OC", np.nan, np.nan, np.nan, np.nan],
                                        ["N", np.nan, np.nan, np.nan, np.nan]],
                                       columns=["Target", "R2", "RMSE", "MAE", "RPD"])
            all_fold_metrics.append(nan_metrics)

        # Cleanup memory
        if 'model' in locals(): del model
        if 'y_true_orig' in locals(): del y_true_orig
        if 'y_pred_orig' in locals(): del y_pred_orig

        del X_train_fold_raw, y_train_fold_raw, X_val_fold_raw, y_val_fold_raw
        del X_scaler, y_scaler, X_train_scaled, y_train_scaled
        if model_name in ["SpectralMLP", "Simple1DCNN"]:
            del optimizer, criterion, scheduler, X_train_t, y_train_t, X_val_t, y_val_t
        torch.cuda.empty_cache()

    # --- Model K-Fold Loop End ---
    model_elapsed = time.time() - model_start_time
    print(f"\n--- K-Fold CV for {model_name} COMPLETE. Total Time: {model_elapsed:.1f}s ---")

    if all_fold_metrics:
        model_summary_df = pd.concat(all_fold_metrics)
        model_summary_df["Model"] = model_name
        all_models_metrics_dfs.append(model_summary_df)

# ------------- 6. Final Summary Report (Compare All Models) -------------
print(f"\n=======================================================")
print(f"--- ALL MODELS K-FOLD CV FINAL SUMMARY ---")
print(f"=======================================================")
total_cv_time = time.time() - total_cv_start_time
print(f"Total K-Fold Run Time: {total_cv_time:.1f}s")

if not all_models_metrics_dfs:
    print("[ERROR] No metrics were collected from any model. Aborting.")
    exit()

try:
    final_summary_df = pd.concat(all_models_metrics_dfs)

    final_stats = final_summary_df.groupby(['Model', 'Target'])[['R2', 'RMSE', 'MAE', 'RPD']].agg(['mean', 'std'])

    # Reindex for a clean, consistent output order
    model_order = [m for m in MODELS_TO_RUN if m in final_stats.index.get_level_values('Model')]
    target_order = ["pH", "OC", "N"]
    final_stats = final_stats.reindex(index=pd.MultiIndex.from_product([model_order, target_order]))

    print("\n--- Aggregated CV Statistics (Mean & Std Dev) ---")


    def format_mean_std(mean, std):
        if pd.isna(mean) or pd.isna(std):
            return "N/A (Run Failed)"
        return f"{mean:.4f} (± {std:.4f})"


    formatted_stats = pd.DataFrame(index=final_stats.index)
    for col in ['R2', 'RMSE', 'MAE', 'RPD']:
        formatted_stats[col] = [
            format_mean_std(mean, std) for mean, std in zip(final_stats[(col, 'mean')], final_stats[(col, 'std')])
        ]

    print(formatted_stats.to_string())

    final_summary_df.to_csv(os.path.join(RESULT_DIR, "kfold_metrics_all_models_raw.csv"), index=False)
    final_stats.to_csv(os.path.join(RESULT_DIR, "kfold_metrics_all_models_summary.csv"))

    print("\n✅ K-Fold Baseline Comparison Complete.")
    print(f"   All results saved to: {RESULT_DIR}")

except Exception as e:
    print(f"[ERROR] Failed to aggregate final CV statistics: {e}")
    import traceback

    traceback.print_exc()

print("=======================================================")