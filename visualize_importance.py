# visualize_importance.py
# Purpose: Loads the trained Champion Model (SpectralMLP), calculates, and visualizes
#          Feature Importance (Saliency Map / Gradient Importance).
# Workflow:
# 1. Strictly reproduce the data loading and preprocessing workflow.
# 2. Load the 'best_final_model.pth'.
# 3. Calculate the Mean Absolute Gradient of output (pH, OC, N) with respect to input bands.
# 4. Plot the results with Wavelength (nm) on X-axis.

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Import Core Modules (Ensure Consistency) ---
from model import SpectralMLP
from data_utils import get_data_splits, augment_data
from utils import seed_everything

# ------------- 1. Configuration (Relative Paths) -------------
seed_everything(42)

# Global relative path definitions
CSV_PATH = "cleaned_soil_data.csv"
RESULT_ROOT = "results"

# 1a. Model Loading Path (Must point to the run directory where the model was saved)
RUN_NAME = "train_final_evaluation"
MODEL_ROOT_DIR = os.path.join(RESULT_ROOT, RUN_NAME)
MODEL_PATH = os.path.join(MODEL_ROOT_DIR, "best_final_model.pth")

# 1b. Image Output Path (Target: results/ root folder)
IMAGE_OUTPUT_DIR = RESULT_ROOT

# Data and Model Parameters must match the training script
PREPROCESS = 'snv_sg_smooth'
SG_WINDOW = 15
SG_POLYORDER = 3
SG_DERIV = 1
USE_AUGMENTATION = False
TEST_SET_RATIO = 0.2
FINAL_VAL_RATIO = 0.15

# Champion Model Parameters
CHAMPION_PARAMS_MLP = {
    'embed_dim': 128,
    'depth': 2,
    'dropout_rate': 0.0,
}

# Visualization Parameters
TARGETS = ["pH", "OC", "N"]
N_SAMPLES_FOR_VIZ = 500  # Number of samples to use for gradient calculation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"--- Running Feature Importance Visualization ---")
print(f"Loading champion model from: {MODEL_PATH}")
print(f"Using device: {DEVICE}")

# ------------- 2. Strictly Reproduce Data Loading and Preprocessing Workflow -------------
try:
    print("Loading all data splits...")
    # Load 80% development set and 20% test set
    X_train_val_raw, y_train_val_raw, X_test_raw, y_test_raw = get_data_splits(
        CSV_PATH, preprocess_method=PREPROCESS, test_size=TEST_SET_RATIO,
        sg_window=SG_WINDOW, sg_polyorder=SG_POLYORDER, sg_deriv=SG_DERIV
    )

    print(f"Splitting 80% set into final train/val (Ratio={FINAL_VAL_RATIO})...")
    # This split isolates the data used to fit the scaler during training
    X_train_final_raw, X_val_final_raw, y_train_final_raw, y_val_final_raw = train_test_split(
        X_train_val_raw, y_train_val_raw, test_size=FINAL_VAL_RATIO, random_state=42
    )

    # Determine data used to fit the scaler (matches final_evaluation script)
    if USE_AUGMENTATION:
        X_train_to_fit_on, y_train_to_fit_on = augment_data(
            X_train_final_raw, y_train_final_raw, random_state=42
        )
    else:
        X_train_to_fit_on, y_train_to_fit_on = X_train_final_raw, y_train_final_raw

    print("Fitting scalers ONLY on final train set (to match training)...")
    X_scaler = StandardScaler().fit(X_train_to_fit_on)

    print("Transforming 20% Independent Test Set...")
    X_test_scaled = X_scaler.transform(X_test_raw)

    # Randomly sample N examples for visualization
    if N_SAMPLES_FOR_VIZ > 0 and N_SAMPLES_FOR_VIZ < len(X_test_scaled):
        print(f"Sampling {N_SAMPLES_FOR_VIZ} test examples...")
        # [!!!] Use explicit RNG for guaranteed reproducibility of visualization samples
        rng = np.random.default_rng(42)
        indices = rng.choice(len(X_test_scaled), N_SAMPLES_FOR_VIZ, replace=False)
        X_test_subset = X_test_scaled[indices]
    else:
        print("Using all test examples...")
        X_test_subset = X_test_scaled

    # Convert to PyTorch Tensor, and set requires_grad = True
    X_test_t = torch.tensor(X_test_subset, dtype=torch.float32, device=DEVICE)
    X_test_t.requires_grad = True  # CRITICAL: Tells PyTorch to calculate gradients for the input

    input_dim = X_test_t.shape[1]
    output_dim = len(TARGETS)

    print(f"   Data ready. Input tensor shape: {X_test_t.shape}")

except Exception as e:
    print(f"[ERROR] Failed to load or preprocess data: {e}")
    import traceback;

    traceback.print_exc()
    exit()

# ------------- 3. Load Trained Champion Model -------------
try:
    print("Loading model structure (SpectralMLP)...")
    model = SpectralMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        embed_dim=CHAMPION_PARAMS_MLP['embed_dim'],
        depth=CHAMPION_PARAMS_MLP['depth'],
        dropout=CHAMPION_PARAMS_MLP['dropout_rate']
    ).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        print(f"[INFO] Please run 'train_final_evaluation.py' first to generate the model file in {MODEL_ROOT_DIR}.")
        exit()

    # Load model weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()  # Set model to evaluation mode
    print("   Model loaded successfully.")

except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit()

# ------------- 4. Calculate Gradients (Saliency) -------------
print("Calculating gradients (Feature Importance)...")

saliency_maps = {}

for i, target_name in enumerate(TARGETS):
    print(f"   Processing target: {target_name} (Output Index {i})...")

    # Zero out previous gradients
    if X_test_t.grad is not None:
        X_test_t.grad.zero_()
    model.zero_grad()

    # 1. Forward pass
    predictions = model(X_test_t)

    # 2. Sum the output of the current target across all N samples
    target_output_sum = predictions[:, i].sum()

    # 3. Backward pass: Calculate Sum(Output[i]) / d(Input[j])
    target_output_sum.backward(retain_graph=True)

    # 4. Get gradient and take the absolute value
    saliency = X_test_t.grad.abs().cpu().numpy()

    # 5. Average across all samples to get one (4200,) importance vector
    saliency_maps[target_name] = saliency.mean(axis=0)

print("   Gradient calculation complete.")

# ------------- 5. Visualization (Updated X-axis) -------------
print("Generating plots...")

# --- MODIFICATION START: Logic to handle Wavelengths vs Band Index ---
wavelengths = None

# 1. Priority: If input dimension is exactly 4200, assume LUCAS standard (400-2500 nm)
if input_dim == 4200:
    print("   [INFO] Detected 4200 bands. Automatically mapping to 400-2500 nm wavelength range.")
    wavelengths = np.linspace(400, 2500, input_dim)

# 2. Fallback: Try to read from CSV if dimension doesn't match 4200
if wavelengths is None:
    try:
        df_temp = pd.read_csv(CSV_PATH, nrows=1)
        wave_cols = [col for col in df_temp.columns if col.replace('.', '', 1).isdigit() and len(col) > 1]

        if len(wave_cols) == input_dim:
            print(f"   Successfully loaded {len(wave_cols)} wavelengths from CSV headers.")
            wavelengths = np.array([float(w) for w in wave_cols])
        else:
            print("   [WARN] Wavelength headers do not match input dimension.")
    except Exception as e:
        print(f"   [WARN] Could not parse wavelengths from CSV: {e}")

# 3. Ultimate Fallback: Use Band Index (0..N)
if wavelengths is None:
    print("   [WARN] Defaulting X-axis to Band Index (0, 1, 2...).")
    wavelengths = np.arange(0, input_dim)
# --- MODIFICATION END ---

plt.figure(figsize=(15, 10))
# Title updated to reflect the content
plt.suptitle(f"Feature Importance (Saliency) for SpectralMLP\n(Based on {N_SAMPLES_FOR_VIZ} test samples)", fontsize=16)

for i, target_name in enumerate(TARGETS):
    plt.subplot(len(TARGETS), 1, i + 1)

    importance_vector = saliency_maps[target_name]

    # Plotting against the 'wavelengths' array we generated above
    plt.plot(wavelengths, importance_vector, label=f"Importance for {target_name}", color=f"C{i}")

    plt.ylabel("Mean Absolute Gradient\n(Importance)")
    plt.title(f"Target: {target_name}")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.5)

    # Optional: Add vertical lines for key chemical absorption features mentioned in paper
    # e.g., Water (~1400, ~1900), Organic (~2200)
    # You can uncomment the lines below if you want to highlight them on the plot automatically
    # if input_dim == 4200:
    #     plt.axvline(x=1400, color='gray', linestyle=':', alpha=0.5)
    #     plt.axvline(x=1900, color='gray', linestyle=':', alpha=0.5)
    #     plt.axvline(x=2200, color='gray', linestyle=':', alpha=0.5)

# Updated X-label
plt.xlabel("Wavelength (nm)")

# Save the plot to the desired results root folder
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
save_path = os.path.join(IMAGE_OUTPUT_DIR,
                         "feature_importance_map.jpg")  # Changed to jpg to match your report request, or keep png

try:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Feature importance plot saved successfully to:\n   {save_path}")
except Exception as e:
    print(f"\n[ERROR] Failed to save plot: {e}")

print("\n--- Visualization Finished ---")