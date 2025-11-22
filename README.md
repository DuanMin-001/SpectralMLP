# SpectralMLP

**Official PyTorch implementation** for the paper:
> **SpectralMLP: A Lightweight and Interpretable Residual MLP for Efficient Soil Property Prediction with Vis–NIR Spectroscopy**

##  Overview
**SpectralMLP** is a lightweight residual Multilayer Perceptron architecture designed to predict soil properties (pH, OC, N) using Visible–Near-Infrared (Vis–NIR) spectroscopy.

Unlike complex Transformer models, SpectralMLP targets the **"optimal complexity sweet spot"**. It delivers high predictive accuracy statistically equivalent to Transformers but significantly reduces computational costs, making it ideal for deployment on portable agricultural devices.

##  Key Highlights
- **High Efficiency:** **37.8% faster** training speed compared to Spectral Transformer.
- **Lightweight:** **16.5% reduction** in parameter count.
- **Robustness:** Proven stability across varying random seeds and soil heterogeneity.
- **Interpretability:** Includes gradient-based saliency analysis consistent with soil chemical mechanisms.

##  Results Summary
Performance comparison based on 5-Fold Cross-Validation (Time averaged over 3 independent runs):

| Model | Parameters | Param. Reduction | Mean R² | Mean RPD | Training Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SpectralTransformer** | 803,331 | — | 0.948 | 4.405 | 82.73 |
| **SpectralMLP (Ours)** | **670,723** | **↓ 16.5%** | **0.947** | **4.378** | **51.44 (↓ 37.8%)** |

> On the independent test set (20%), SpectralMLP achieved **Excellent** prediction levels (RPD > 4.0) for all targets.

## 🛠️ Getting Started

### 1. Requirements

This project relies on standard Python data science and deep learning libraries.

Save the dependencies listed below into a `requirements.txt` file and run:

```bash
pip install -r requirements.txt
```

### 2. Data Preparation (Crucial)

This project uses the **LUCAS 2009 Topsoil Dataset**.

1. Download the raw dataset and save it as `LUCAS.SOIL_corr.csv` in the project root directory.

2. Run the preparation script to generate the cleaned dataset:

   Bash

   ```
   python prepare_data.py
   ```

   *Output: `cleaned_soil_data.csv` will be generated.*



##  Usage Guide

You can reproduce the experiments in the following order:

### Step 1: Hyperparameter Tuning (Grid Search)

Optimize parameters for SpectralMLP and the 1D-CNN baseline.

```
python train_opt_grid_mlp.py
python train_opt_grid_cnn.py
```

### Step 2: Ablation Study (Table 1)

Verify the impact of preprocessing strategies, loss functions, and model depth.

```
python train_ablation.py
```

*This script reproduces the "Ablation Study" section (Table 1) of the manuscript.*

### Step 3: K-Fold Comparison (Table 2 & 3)

Run 5-Fold Cross-Validation to compare SpectralMLP against Transformers and traditional baselines (PLS, SVR, RF, CNN).

```
# Efficiency comparison (SpectralMLP vs Transformer)
python train_kfold_mlpVStransV1.py

# Baseline comparison (SpectralMLP vs All Baselines)
python train_kfold_baselineVS.py
```

### Step 4: Statistical Analysis

Generate p-values for significance testing (paired t-test).

> ** Important Pre-requisite:**
> Before running the analysis, you must **manually merge** the raw result files generated in Step 2 into a single CSV file.
>
> 1. Take `kfold_metrics_summary_raw.csv` from `results/train_kfold_mlpVStransV1/`.
> 2. Take `kfold_metrics_all_models_raw.csv` from `results/train_kfold_baselineVS/`.
> 3. Merge them into one file named **`all_models_raw.csv`**.
> 4. Place this file in the folder: `results/statistical_analysis/` (create the folder if it doesn't exist).

Once the data is ready, run:

```
python statistical_analysis.py
```

### Step 5: Final Evaluation & Visualization

Evaluate the best model on the independent test set and generate figures.

```
# Final blind test (Table 5)
python train_final_evaluation.py

# Stability test for 1D-CNN (Table 4)
python cnn_final_stability_test_automated.py

# Generate Preprocessing Figure (Fig 1) & Feature Importance Map (Fig 4)
python visualize_preprocessing.py
python visualize_importance.py
```

##  Project Structure
- `model.py`: Definitions for SpectralMLP and SpectralTransformerMLP.
- `baselines.py`: Definitions for PLS, SVR, RF, and Simple1DCNN.
- `data_utils.py`: Data loading, SNV/SG preprocessing, and augmentation logic.
- `utils.py`: Metrics calculation, seed management, and plotting functions.
- `results/`: All logs, plots, and CSV reports are saved here.

## License
This project is licensed under the terms of the MIT License. See the LICENSE file for full details.
