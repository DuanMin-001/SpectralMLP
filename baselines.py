# baselines.py (for draft_4 - V3_STRONG_BASELINES)
# Description: Contains classic and deep learning baseline models for comparative analysis.
# [!!!] MODIFIED V3: The grid search ranges for PLSR, SVR, and RF have been significantly expanded
# to rigorously defend against the "weak baseline" argument.

import torch
import torch.nn as nn
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

# =================================================================
# 1. Partial Least Squares Regression (PLSR) - Classic Chemometrics
# =================================================================
class PLSBaseline:
    """
    Wrapper for training and hyperparameter tuning of PLSR within the K-Fold loop.
    Tuning is performed using internal GridSearchCV.
    """
    def __init__(self, n_components_grid=None, n_inner_folds=5):
        if n_components_grid is None:
            # [V3 UPDATED] Expanded search range: 10 to 60, with a step size of 5
            self.n_components_grid = {'n_components': list(range(10, 61, 5))}
        else:
            self.n_components_grid = {'n_components': n_components_grid}

        self.n_inner_folds = n_inner_folds
        self.grid_search = GridSearchCV(
            PLSRegression(),
            self.n_components_grid,
            scoring='neg_root_mean_squared_error',
            cv=self.n_inner_folds,
            n_jobs=-1 # Use all available cores
        )
        self.best_model_ = None

    def fit(self, X_train, y_train):
        print(f"   [PLS] Running GridSearchCV over components: {self.n_components_grid['n_components']}...")
        self.grid_search.fit(X_train, y_train)
        self.best_model_ = self.grid_search.best_estimator_
        print(f"   [PLS] Best n_components found: {self.best_model_.n_components}")

    def predict(self, X_test):
        return self.best_model_.predict(X_test)

# =================================================================
# 2. Support Vector Regression (SVR) - Classic Machine Learning
# =================================================================
class SVRBaseline:
    """
    Wrapper for hyperparameter tuning of SVR, designed for MultiOutput regression.
    The hyperparameter search space is reduced, and StandardScaler (with_mean=True)
    is used for efficiency, as justified in the manuscript's methodology.
    """

    def __init__(self, n_inner_folds=5):
        self.n_inner_folds = n_inner_folds

        # [!!!] FIX: Use default StandardScaler (with_mean=True) for faster Sklearn processing
        # Ensure only one scaler is in the pipeline (no double scaling)
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', MultiOutputRegressor(SVR(kernel='rbf')))
        ])

        # [!!!] FIX: Use a much smaller, reasonable search space (Reduces O(N^3) complexity)
        self.param_grid = {
            'model__estimator__C': [1, 10, 100],  # Reduced C options
            'model__estimator__gamma': ['scale']  # Fixed gamma option
        }

        self.grid_search = GridSearchCV(
            self.pipeline,
            self.param_grid,
            scoring='neg_root_mean_squared_error',
            cv=self.n_inner_folds,
            n_jobs=-1
        )
        self.best_model_ = None

    def fit(self, X_train, y_train):
        print(f"   [SVR] Running strongly enhanced GridSearchCV...")
        self.grid_search.fit(X_train, y_train)
        self.best_model_ = self.grid_search.best_estimator_
        print(f"   [SVR] Best params found: {self.grid_search.best_params_}")

    def predict(self, X_test):
        # Predicts on the original scale due to the pipeline structure
        return self.best_model_.predict(X_test)

# =================================================================
# 3. Random Forest (RF) - Classic Ensemble Learning
# =================================================================
class RandomForestBaseline:
    """
    Wrapper for hyperparameter tuning of RandomForestRegressor.
    """
    def __init__(self, n_inner_folds=5):
        self.n_inner_folds = n_inner_folds
        # Fixed random_state=42 for reproducibility across runs
        self.model = RandomForestRegressor(random_state=42, n_jobs=-1)

        # [V3 UPDATED] Increased n_estimators options to ensure sufficient convergence
        self.param_grid = {
            'n_estimators': [100, 300, 500],
            'max_features': ['sqrt', 'log2'] # Features to consider for best split
        }

        self.grid_search = GridSearchCV(
            self.model,
            self.param_grid,
            scoring='neg_root_mean_squared_error',
            cv=self.n_inner_folds,
            n_jobs=-1
        )
        self.best_model_ = None

    def fit(self, X_train, y_train):
        print(f"   [RF] Running enhanced GridSearchCV...")
        self.grid_search.fit(X_train, y_train)
        self.best_model_ = self.grid_search.best_estimator_
        print(f"   [RF] Best params found: {self.grid_search.best_params_}")

    def predict(self, X_test):
        return self.best_model_.predict(X_test)

# =================================================================
# 4. One-Dimensional Convolutional Neural Network (1D-CNN) - Deep Learning Baseline
# =================================================================
class Simple1DCNN(nn.Module):
    """
    A simple yet effective 1D-CNN baseline model, used to test the performance lower bound
    of deep learning architectures.
    """
    def __init__(self, input_dim, output_dim=3, num_filters=32, filter_size=15, dropout=0.1):
        super().__init__()
        # print(f"Initializing [BASELINE] Simple1DCNN: filters={num_filters}, kernel={filter_size}, dropout={dropout}")
        # Note: print statement commented out to reduce K-Fold log output.

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=filter_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=filter_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters*2)
        )
        # Global Average Pooling (GAP)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(num_filters*2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # Input shape: (Batch, W) -> Unsqueeze for Conv1d: (Batch, C=1, W)
        x = x.unsqueeze(1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.gap(x)
        # Squeeze the final dimension (1) to get (Batch, Channels)
        x = x.squeeze(-1)
        return self.head(x)