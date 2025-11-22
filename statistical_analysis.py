# statistical_analysis.py
# Purpose:
#   1. Load the merged final analysis table (K-Fold results).
#   2. Perform paired t-tests between SpectralMLP and all other baseline models.
#   3. Output the p-value matrices used for Table 3 in the manuscript.

import pandas as pd
from scipy.stats import ttest_rel
import os
import itertools

# --- 1. Configuration (Relative Paths) ---
RESULT_ROOT = "results"
ANALYSIS_RUN_NAME = "statistical_analysis"
ANALYSIS_RESULT_DIR = os.path.join(RESULT_ROOT, ANALYSIS_RUN_NAME)
os.makedirs(ANALYSIS_RESULT_DIR, exist_ok=True)

# Base filename expected
BASE_FILENAME = "all_models_raw"

REFERENCE_MODEL = "SpectralMLP"
MODELS_TO_COMPARE = [
    "SpectralTransformerMLP",
    "PLSBaseline",
    "SVRBaseline",
    "RandomForestBaseline",
    "Simple1DCNN"
]
TARGETS = ["pH", "OC", "N"]
METRICS = ["R2", "RPD", "RMSE", "MAE"]

print(f"--- Running Statistical Significance Analysis (Paired t-test) ---")

# --- 2. Data Loading (Smart Search) ---
# We search for the file in multiple likely locations
possible_paths = [
    # 1. Standard sub-directory
    os.path.join(ANALYSIS_RESULT_DIR, f"{BASE_FILENAME}.csv"),
    os.path.join(ANALYSIS_RESULT_DIR, f"{BASE_FILENAME}.xlsx"),
    # 2. Results root directory
    os.path.join(RESULT_ROOT, f"{BASE_FILENAME}.csv"),
    os.path.join(RESULT_ROOT, f"{BASE_FILENAME}.xlsx"),
    # 3. Project root directory (same as script)
    f"{BASE_FILENAME}.csv",
    f"{BASE_FILENAME}.xlsx"
]

df_raw = None
found_path = ""

try:
    for path in possible_paths:
        if os.path.exists(path):
            found_path = path
            print(f"   -> Found data file at: {found_path}")
            if path.endswith('.csv'):
                df_raw = pd.read_csv(path)
            else:
                df_raw = pd.read_excel(path, engine='openpyxl')
            break

    if df_raw is None:
        # Explicitly raise error if loop finishes without finding file
        raise FileNotFoundError(
            f"Could not find '{BASE_FILENAME}.csv' or .xlsx in any standard locations.\n"
            f"Checked: results/statistical_analysis/, results/, and project root."
        )

    print(f"   -> Successfully loaded {len(df_raw)} rows.")

    # [Robustness Check] Automatically infer/create the 'Fold' column if missing
    if 'Fold' not in df_raw.columns:
        print("   (Detected missing 'Fold' column. Attempting to autofill Folds 1-5...)")

        if 'Target' not in df_raw.columns or 'Model' not in df_raw.columns:
            print("[ERROR] Your raw file lacks 'Target' or 'Model' columns. Cannot proceed.")
            exit()

        # Group by 'Model' and 'Target' and number the rows within each group
        df_raw['Fold'] = df_raw.groupby(['Model', 'Target']).cumcount() + 1
        print("   (Fold column successfully created)")

    # Ensure no duplicated entries
    df_raw = df_raw.drop_duplicates(subset=['Model', 'Fold', 'Target'])

except FileNotFoundError as e:
    print(f"[ERROR] {e}")
    exit()
except Exception as e:
    print(f"[ERROR] Error loading data: {e}")
    exit()

# Final validation of reference model existence
if REFERENCE_MODEL not in df_raw['Model'].unique():
    print(f"[ERROR] Reference model not found in the data: '{REFERENCE_MODEL}'")
    print(f"Available models in CSV: {df_raw['Model'].unique()}")
    print("Please check if the model names in your CSV match the code (e.g., 'SpectralMLP').")
    exit()

# --- 3. Execute Paired t-test ---

results = []

for target in TARGETS:
    for metric in METRICS:
        try:
            # 1. Get the 5 Fold scores for the REFERENCE MODEL (SpectralMLP)
            ref_scores = df_raw[
                (df_raw['Model'] == REFERENCE_MODEL) &
                (df_raw['Target'] == target)
                ].sort_values(by="Fold")[metric].values

            # Validation checks
            if len(ref_scores) == 0:
                continue
            if len(ref_scores) < 5:
                print(f"[WARN] {REFERENCE_MODEL} (Target={target}) scores incomplete (N={len(ref_scores)}), skipping.")
                continue

            for model_to_compare in MODELS_TO_COMPARE:
                if model_to_compare not in df_raw['Model'].unique():
                    continue

                # 2. Get the 5 Fold scores for the COMPARISON MODEL
                comp_scores = df_raw[
                    (df_raw['Model'] == model_to_compare) &
                    (df_raw['Target'] == target)
                    ].sort_values(by="Fold")[metric].values

                if len(comp_scores) != len(ref_scores):
                    print(
                        f"[WARN] {model_to_compare} ({target}) count mismatch ({len(comp_scores)} vs {len(ref_scores)}). Skipping.")
                    continue

                # 3. Perform Paired t-test
                t_stat, p_value = ttest_rel(ref_scores, comp_scores)

                results.append({
                    "Target": target,
                    "Metric": metric,
                    "Model_A": REFERENCE_MODEL,
                    "Model_B": model_to_compare,
                    "p_value": p_value,
                    "A_mean": ref_scores.mean(),
                    "B_mean": comp_scores.mean(),
                })

        except Exception as e:
            print(f"[ERROR] Paired t-test failed for {target} - {metric}: {e}")

# --- 4. Format Output p-value Report ---
print("\n" + "=" * 80)
print(f"--- Statistical Analysis Report (p-value) ---")
print(f"---   {REFERENCE_MODEL} vs. Other Baselines   ---")
print("=" * 80)
print("(p < 0.05 indicates a statistically significant difference)\n")

if not results:
    print("[WARN] No t-test results generated. Please check your input CSV content.")
    exit()

df_results = pd.DataFrame(results)

# Focus on RPD
print("--- Key Metric: RPD (p-values) ---")
pivot_rpd = df_results[df_results['Metric'] == 'RPD'].pivot_table(
    index='Model_B', columns='Target', values='p_value'
)
print(pivot_rpd.to_string(float_format="%.4f"))

# Focus on R2
print("\n--- Key Metric: R2 (p-values) ---")
pivot_r2 = df_results[df_results['Metric'] == 'R2'].pivot_table(
    index='Model_B', columns='Target', values='p_value'
)
print(pivot_r2.to_string(float_format="%.4f"))

# Focus on RMSE
print("\n--- Key Metric: RMSE (p-values) ---")
pivot_rmse = df_results[df_results['Metric'] == 'RMSE'].pivot_table(
    index='Model_B', columns='Target', values='p_value'
)
print(pivot_rmse.to_string(float_format="%.4f"))

print("\n" + "=" * 80)
print("--- How to Interpret this Report ---")
print("1. p-value < 0.05: The performance difference is statistically significant.")
print("2. p-value > 0.05: The performance difference is NOT statistically significant.")
print("=" * 80)

# Save the full results DataFrame
summary_save_path = os.path.join(ANALYSIS_RESULT_DIR, "statistical_analysis_p_values.csv")
try:
    df_results.to_csv(summary_save_path, index=False, float_format='%.6f')
    print(f"\n✅ Complete p-value report saved to: {summary_save_path}")
except Exception as e:
    print(f"\n[ERROR] Failed to save p-value report: {e}")