import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
import os
import random

# Note: Removed the definition of seed_everything to prevent redundancy.

# --- Path Configuration (Updated to Relative Path) ---
DEFAULT_CSV_PATH = "cleaned_soil_data.csv"


# --- Spectral Preprocessing Functions ---
def apply_snv(X):
    """
    Applies Standard Normal Variate (SNV) transformation to the spectra.
    SNV removes multiplicative interference (scatter) and baseline variations.

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
    Applies Savitzky-Golay (SG) filtering for smoothing or calculating derivatives.

    Args:
        X (pd.DataFrame or np.ndarray): Spectral data.
        window (int): The length of the filter window (must be odd).
        polyorder (int): The order of the polynomial used to fit the samples.
        deriv (int): The order of the derivative to compute (0=smoothing, 1=1st derivative).

    Returns:
        np.ndarray: SG filtered/derived spectra.
    """
    if isinstance(X, pd.DataFrame): X = X.values

    if X.shape[1] < window:
        print(f"[WARN] SG: Spectral length ({X.shape[1]}) < window ({window}). Returning original data.")
        return X

    if deriv > polyorder and deriv > 0:
        print(
            f"[WARN] SG: Derivative order ({deriv}) > Polynomial order ({polyorder}). This is generally not recommended.")

    try:
        return savgol_filter(X, window_length=window, polyorder=polyorder, deriv=deriv, axis=1, mode='interp')
    except Exception as e:
        print(f"[ERROR] SG Filtering failed (w={window}, p={polyorder}, d={deriv}): {e}. Returning original data.")
        return X


# --- Internal Core Loading and Cleaning Function ---
def _load_and_clean_raw(csv_path, preprocess_method='snv_sg_smooth', sg_window=15, sg_polyorder=3, sg_deriv=1):
    """
    Internal function to load pre-cleaned data and perform final data preparation steps:
    1. Load data from the cleaned CSV file.
    2. Apply spectral preprocessing (SNV, SG).
    3. Perform final statistical cleaning (imputation and clipping).

    Returns:
        tuple: (X_clean, y_clean) - Unaugmented, unscaled NumPy arrays for features and targets.
    """
    if not os.path.exists(csv_path): raise FileNotFoundError(f"Data file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path);
        print(f">> Successfully loaded data: {csv_path}")
    except Exception as e:
        raise IOError(f"Error reading CSV file: {csv_path} - {e}")

    target_cols = ['pH.in.CaCl2', 'OC', 'N']
    X_df = df.drop(columns=target_cols, errors='ignore')
    y_df = df[target_cols]
    if not all(col in df.columns for col in target_cols): raise ValueError(
        f"CSV file must contain columns: {target_cols}")

    # 1. Spectral Preprocessing
    X_processed = X_df.apply(pd.to_numeric, errors='coerce').values
    print(f">> Initial X shape: {X_processed.shape}")

    if preprocess_method == 'snv':
        print(">> Applying SNV preprocessing...")
        X_processed = apply_snv(X_processed)
    elif preprocess_method == 'snv_sg_smooth':
        print(">> Applying SNV...")
        X_snv = apply_snv(X_processed);
        print(f">> Followed by SG (w={sg_window}, p={sg_polyorder}, d={sg_deriv})...")
        X_processed = apply_sg_smooth(X_snv,
                                      window=sg_window,
                                      polyorder=sg_polyorder,
                                      deriv=sg_deriv)
    elif preprocess_method is None:
        print(">> No spectral preprocessing performed.")
    else:
        print(f"[WARN] Unknown preprocessing: '{preprocess_method}'. Skipping.")

    # Check for NaN/Inf introduced during preprocessing
    if not np.all(np.isfinite(X_processed)):
        print("[WARN] NaN/Inf detected after preprocessing, replacing with 0.");
        X_processed = np.nan_to_num(X_processed)

    # 2. Statistical Cleaning (Median Imputation and 1%/99% clip)
    X_processed_df = pd.DataFrame(X_processed, columns=X_df.columns)
    y_df_numeric = y_df.apply(pd.to_numeric, errors='coerce')

    # Calculate median for imputation
    X_median = X_processed_df.median();
    y_median = y_df_numeric.median()

    # Impute missing values with median
    X_filled = X_processed_df.fillna(X_median);
    y_filled = y_df_numeric.fillna(y_median)

    # Apply 1%/99% Quantile Clipping to features
    for col in X_filled.select_dtypes(include=np.number).columns:
        lower, upper = X_filled[col].quantile(0.01), X_filled[col].quantile(0.99);
        X_filled[col] = X_filled[col].clip(lower, upper)

    # Apply 1%/99% Quantile Clipping to targets
    for col in y_filled.select_dtypes(include=np.number).columns:
        lower, upper = y_filled[col].quantile(0.01), y_filled[col].quantile(0.99);
        y_filled[col] = y_filled[col].clip(lower, upper)

    X_clean = X_filled.values;
    y_clean = y_filled.values

    print(f">> Data loading and preprocessing complete. Final X shape: {X_clean.shape}, y shape: {y_clean.shape}")
    return X_clean, y_clean


# --- Helper: Data Augmentation ---
def augment_data(X_train, y_train, random_state=42):
    """
    Applies simple additive noise and multiplicative scaling for data augmentation
    on the training set only.

    Returns: Stacked augmented dataset.
    """
    print(">> Applying data augmentation...")
    np.random.seed(random_state)

    def augment_features(X_np):
        noise = np.random.normal(0, 0.01, X_np.shape);
        scale = np.random.uniform(0.98, 1.02, X_np.shape);
        return X_np * scale + noise

    X_aug = augment_features(X_train);
    X_train_augmented = np.vstack((X_train, X_aug));
    y_train_augmented = np.vstack((y_train, y_train))
    print(f">> Augmented training set shape: X={X_train_augmented.shape}, y={y_train_augmented.shape}")
    return X_train_augmented, y_train_augmented


# --- Helper: Standardization ---
def scale_data(X_train, y_train, X_val, y_val):
    """
    Fits StandardScaler only on the training data, then transforms all splits (train/val).

    Returns: Scaled arrays and the fitted scalers.
    """
    X_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)

    X_train_scaled = X_scaler.transform(X_train)
    y_train_scaled = y_scaler.transform(y_train)
    X_val_scaled = X_scaler.transform(X_val)
    y_val_scaled = y_scaler.transform(y_val)

    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_scaler, y_scaler


# --- Core Data Loading Function ---
def get_data_splits(csv_path=DEFAULT_CSV_PATH, preprocess_method='snv_sg_smooth', test_size=0.2, random_state=42,
                    **kwargs):
    """
    (***Core Data Function***)
    Loads, preprocesses, cleans, and performs the single, unique Train/Val + Test split (e.g., 80/20).
    This split is fixed and serves as the foundation for K-Fold runs and the final blind test.

    Args:
        csv_path (str): Path to the cleaned data CSV file.
        preprocess_method (str): Spectral preprocessing method to apply.
        test_size (float): Proportion of the dataset for the independent test split (default 0.2).
        random_state (int): Seed used to guarantee reproducibility of the train/test split.
        **kwargs: Optional arguments for SG filter (window, polyorder, deriv).

    Returns:
        tuple: (X_train_val_raw, y_train_val_raw): 80% data for K-Fold/tuning.
               (X_test_raw, y_test_raw): 20% data for final blind test.
    """
    sg_window = kwargs.get('sg_window', 15)
    sg_polyorder = kwargs.get('sg_polyorder', 3)
    sg_deriv = kwargs.get('sg_deriv', 1)
    print(f"\n--- Loading Data and Performing Fixed Train/Test Split (Test Size={test_size}) ---")

    # 1. Load all processed and cleaned data
    X_clean, y_clean = _load_and_clean_raw(
        csv_path,
        preprocess_method=preprocess_method,
        sg_window=sg_window,
        sg_polyorder=sg_polyorder,
        sg_deriv=sg_deriv
    )

    # 2. Perform the one-time Train/Val + Test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_clean, y_clean, test_size=test_size, random_state=random_state
    )

    print(f"   Split complete:")
    print(f"   Train/Validation Set (80%): X={X_train_val.shape}, y={y_train_val.shape}")
    print(f"   Independent Test Set (20%): X={X_test.shape}, y={y_test.shape}")
    print("   [INFO] 'Test Set' is held out. All subsequent development should use the 'Train/Validation Set'.")

    return X_train_val, y_train_val, X_test, y_test