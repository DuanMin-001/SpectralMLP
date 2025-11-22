# cnn_parameter_est.py (1D-CNN Parameter Count Statistics)
# Purpose: Utility script to calculate the total number of trainable parameters
#          for the optimized Simple1DCNN baseline.

import torch
from baselines import Simple1DCNN

# ----------------------------------------------------
# 1. Configuration of Optimal 1D-CNN Parameters
# (Verify these values against your cnn_grid_search_summary.csv)
# ----------------------------------------------------
INPUT_DIM = 4200  # Number of spectral bands
OUTPUT_DIM = 3    # Targets: pH, OC, N
NUM_FILTERS = 32  # Optimal filters from grid results
FILTER_SIZE = 15  # Optimal kernel size from grid results
DROPOUT = 0.1     # Optimal dropout rate from grid results

# 2. Instantiate Model and Calculate Parameters
try:
    model = Simple1DCNN(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        num_filters=NUM_FILTERS,
        filter_size=FILTER_SIZE,
        dropout=DROPOUT
    )

    # Calculate total number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 50)
    print(f"1D-CNN (Optimal Baseline) Parameter Count:")
    print(f"Configuration: Filters={NUM_FILTERS}, Kernel={FILTER_SIZE}")
    print(f"Total Trainable Parameters: {total_params:,}")
    print("=" * 50)
    print("Please use this number in your final manuscript.")

except ImportError:
    print("[ERROR] Could not import 'baselines.py'. Ensure the script is in the same directory.")
except Exception as e:
    print(f"An error occurred: {e}")