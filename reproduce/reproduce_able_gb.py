# -*- coding: utf-8 -*-
"""
GB_3200s.py — ABLE 3200s dissociation SPR data: definitive Gradient Boosting model.

Reproduces the OPTIMAL hyperparameters found by the full grid search in step 1
(Gradient_boosting_3200s.py, 200,000 combos × 7 seeds). Single best model:

    Seed #2   split random_state = 123,   model random_state = 2
    Train R² = 0.9525    Test R² = 0.8707    R² diff = 0.0819
    (3200s baseline: Test R² = 0.6758, gap = 0.258  →  gap closed to 0.082)

Data     : Parameters_3200s.xlsx  (133 raw rows → 102 after cleaning)
Target   : ln(K_D^bipara)
Features : 13 engineered descriptors (see predictor_names below)

All visualisations + model artefacts are written to a dedicated folder
(GB_3200s_output/) so the step-1 grid-search tree stays untouched.
"""

import os
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = _HERE
INPUT_FILE = os.path.join(DATA_DIR, 'data', 'Parameters_3200s.xlsx')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Nature 5-colour palette (fixed, assigned per series)
# ---------------------------------------------------------------------------
C_BLUE = '#457b9d'
C_RED = '#e63946'
C_GREEN = '#2a9d8f'
C_GOLD = '#e9c46a'
C_GRAY = '#6b7280'

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.transparent': True,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'axes.titlesize': 11,
})

# ---------------------------------------------------------------------------
# Column mapping (exact source headers, incl. full-width close parens)
# ---------------------------------------------------------------------------
EXCEL_COLUMNS = {
    'K_D^N':        "KD,N (nM)",
    'K_D^C':        "KD,C (nM)",
    'K_D^bipara':   "KD,Bipara (nM)",
    'd_epi':        "depi (Å)",
    'L_link':       "Llink (Å）",
    'D_mono':       "Dmono (Å）",
    'R_link':       "Rlink",
    'R_area':       "Rarea",
    'A_N':          "Aarea,N",
    'A_C':          "Aarea,C",
    'n_t^N':        "nt,N",
    'n_t^C':        "nt,C",
    'R_bond':       "Rbond",
    'S_geometry':   "Sgeometry",
}

PREDICTOR_NAMES = [
    'K_D^N', 'K_D^C', 'D_mono', 'd_epi',
    'L_link', 'R_link', 'R_area', 'R_bond',
    'A_N', 'A_C', 'n_t^N', 'n_t^C', 'S_geometry',
]

# LaTeX-safe labels for plotting (no Unicode sub/superscripts)
FEATURE_LABELS = {
    'K_D^N':      r'$K_D^{N}$',
    'K_D^C':      r'$K_D^{C}$',
    'D_mono':     r'$D_{mono}$',
    'd_epi':      r'$d_{epi}$',
    'L_link':     r'$L_{link}$',
    'R_link':     r'$R_{link}$',
    'R_area':     r'$R_{area}$',
    'R_bond':     r'$R_{bond}$',
    'A_N':        r'$A_N$',
    'A_C':        r'$A_C$',
    'n_t^N':      r'$n_{t}^{N}$',
    'n_t^C':      r'$n_{t}^{C}$',
    'S_geometry': r'$S_{geometry}$',
}

# ---------------------------------------------------------------------------
# Optimal hyperparameters (seed #2 rank 1)
# ---------------------------------------------------------------------------
SPLIT_RANDOM_STATE = 123   # train/test split seed (seed #2)
MODEL_RANDOM_STATE = 2     # GradientBoostingRegressor random_state

BEST_PARAMS = {
    'loss': 'huber',
    'learning_rate': 0.3,
    'n_estimators': 150,
    'subsample': 0.8,
    'max_features': 0.6,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'min_impurity_decrease': 0.001,
    'ccp_alpha': 0.005,
    'validation_fraction': 0.1,
    'n_iter_no_change': 10,
    'tol': 0.0001,
}

# Expected metrics (for reproducibility self-check)
EXPECTED_TRAIN_R2 = 0.952549171171184
EXPECTED_TEST_R2 = 0.8706819762478113


# ---------------------------------------------------------------------------
# Data loading + cleaning (identical to Gradient_boosting_3200s.py)
# ---------------------------------------------------------------------------
def load_and_prepare():
    df_raw = pd.read_excel(INPUT_FILE)
    required = list(EXCEL_COLUMNS.values())
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise SystemExit(f"Missing columns in Excel: {missing}")

    df = df_raw[required].copy()
    df.columns = list(EXCEL_COLUMNS.keys())

    kd_cols = ['K_D^N', 'K_D^C', 'K_D^bipara']
    for col in kd_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()

    for col in kd_cols:
        if (df[col] <= 0).any():
            raise ValueError(f"Column {col} contains <=0 values")
        df[col] = np.log(df[col])

    X = df[PREDICTOR_NAMES].values
    y = df['K_D^bipara'].values
    return df, X, y


# ---------------------------------------------------------------------------
# Split (binned stratified, identical to search script)
# ---------------------------------------------------------------------------
def make_split(X, y):
    n_samples = len(y)
    n_bins = min(10, max(4, n_samples // 10))
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    y_binned = discretizer.fit_transform(y.reshape(-1, 1)).flatten().astype(int)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SPLIT_RANDOM_STATE)
    train_index, test_index = next(iter(sss.split(X, y_binned)))
    return train_index, test_index


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_predicted_vs_experimental(y_train, y_train_pred, y_test, y_test_pred, out_dir):
    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    ax.scatter(y_train, y_train_pred, color=C_BLUE, s=30, alpha=0.65,
               edgecolor='none', label=f'Train (n={len(y_train)})')
    ax.scatter(y_test, y_test_pred, color=C_RED, s=42, alpha=0.85,
               edgecolor='none', marker='^', label=f'Test (n={len(y_test)})')

    lo = min(y_train.min(), y_test.min(), y_train_pred.min(), y_test_pred.min())
    hi = max(y_train.max(), y_test.max(), y_train_pred.max(), y_test_pred.max())
    ax.plot([lo, hi], [lo, hi], ls='--', color=C_GRAY, lw=0.9, zorder=0)

    ax.set_xlabel(r'Experimental $\ln(K_D^{\mathrm{bipara}})$')
    ax.set_ylabel(r'Predicted $\ln(K_D^{\mathrm{bipara}})$')
    ax.legend(loc='upper left', frameon=False)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'fig1_predicted_vs_experimental.{ext}'))
    plt.close(fig)


def fig_residuals(y_train, y_train_pred, y_test, y_test_pred, out_dir):
    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    train_res = y_train - y_train_pred
    test_res = y_test - y_test_pred
    ax.scatter(y_train_pred, train_res, color=C_BLUE, s=30, alpha=0.65,
               edgecolor='none', label=f'Train (n={len(y_train)})')
    ax.scatter(y_test_pred, test_res, color=C_RED, s=42, alpha=0.85,
               edgecolor='none', marker='^', label=f'Test (n={len(y_test)})')
    ax.axhline(0.0, ls='--', color=C_GRAY, lw=0.9, zorder=0)
    ax.set_xlabel(r'Predicted $\ln(K_D^{\mathrm{bipara}})$')
    ax.set_ylabel('Residual')
    ax.legend(loc='upper right', frameon=False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'fig2_residuals.{ext}'))
    plt.close(fig)


def fig_feature_importance(importances, out_dir):
    order = np.argsort(importances)[::-1]
    labels = [FEATURE_LABELS[PREDICTOR_NAMES[i]] for i in order]
    values = importances[order]

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.barh(range(len(values)), values[::-1], color=C_GOLD, edgecolor='none')
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlabel('Feature importance')
    ax.invert_yaxis()
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'fig3_feature_importance.{ext}'))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    df, X, y = load_and_prepare()
    print(f"  Cleaned samples: {len(df)}")

    train_index, test_index = make_split(X, y)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(f"  Train: {len(X_train)}   Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingRegressor(**BEST_PARAMS, random_state=MODEL_RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    r2_diff = abs(train_r2 - test_r2)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print("\n=== GB 3200s optimal model ===")
    print(f"  Train R² = {train_r2:.4f}   RMSE = {train_rmse:.3f}   MAE = {train_mae:.3f}")
    print(f"  Test  R² = {test_r2:.4f}   RMSE = {test_rmse:.3f}   MAE = {test_mae:.3f}")
    print(f"  R² diff  = {r2_diff:.4f}")
    print(f"  Early-stopped trees: {model.n_estimators_} / {BEST_PARAMS['n_estimators']}")

    # Reproducibility self-check
    tol = 1e-6
    ok = abs(train_r2 - EXPECTED_TRAIN_R2) < tol and abs(test_r2 - EXPECTED_TEST_R2) < tol
    print(f"\n  {'OK  REPRODUCED exactly' if ok else 'MISMATCH vs expected'} "
          f"(tol {tol})")

    # Visualisations
    print("\nGenerating figures...")
    fig_predicted_vs_experimental(y_train, y_train_pred, y_test, y_test_pred, OUTPUT_DIR)
    fig_residuals(y_train, y_train_pred, y_test, y_test_pred, OUTPUT_DIR)
    importances = model.feature_importances_
    fig_feature_importance(importances, OUTPUT_DIR)

    # Feature importance CSV
    feat_imp = pd.DataFrame({
        'Feature': PREDICTOR_NAMES,
        'Importance': importances,
    }).sort_values('Importance', ascending=False)
    feat_imp.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)

    # Predictions CSV (for downstream verification)
    pred_df = pd.DataFrame({
        'split': ['train'] * len(y_train) + ['test'] * len(y_test),
        'y_experimental': np.concatenate([y_train, y_test]),
        'y_predicted': np.concatenate([y_train_pred, y_test_pred]),
    })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions.csv'), index=False)

    # Model + scaler artefacts
    with open(os.path.join(OUTPUT_DIR, 'best_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Summary JSON
    summary = {
        'model': 'GradientBoostingRegressor',
        'split_random_state': SPLIT_RANDOM_STATE,
        'model_random_state': MODEL_RANDOM_STATE,
        'parameters': BEST_PARAMS,
        'performance': {
            'train': {'r2': train_r2, 'rmse': train_rmse, 'mae': train_mae},
            'test': {'r2': test_r2, 'rmse': test_rmse, 'mae': test_mae},
            'r2_difference': r2_diff,
        },
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'features': PREDICTOR_NAMES,
        'target': 'ln(K_D^bipara)',
        'reproduced_exactly': bool(ok),
        'generated_at': datetime.now().isoformat(),
    }
    with open(os.path.join(OUTPUT_DIR, 'model_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\nAll outputs written to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
