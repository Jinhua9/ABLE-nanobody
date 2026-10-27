# ABLE: AlphaFold-assisted Biparatopic Nanobody Avidity Prediction

## Overview
This repository contains the implementation of the ABLE model, a gradient boosting regression model for predicting biparatopic nanobody avidity from structural and energetic features, as described in:

**"AlphaFold-assisted high-avidity biparatopic nanobodies designed by ABLE"**

ABLE enables rational design of high-avidity biparatopic nanobodies by predicting apparent dissociation constants (KD) based on 13 structural features derived from AlphaFold-predicted complexes.

**v2.0 (2026):** the repository now ships the **final published model** — the seed-2/rank-1 configuration of the 200,000-combination × 7-seed hyperparameter search (train R² = 0.95, test R² = 0.87) — together with a browser-based prediction web app and an exact-reproduction package. The original v1 model is preserved as `models/ABLE_model_v1_legacy.pkl`.

### What's new in v2.0
- **Updated model** (`models/ABLE_model.pkl`): final published gradient boosting configuration (see `reproduce/` for the full protocol).
- **Zero-install web predictor** (`docs/`, served via GitHub Pages): enter the 9 raw measurements in a browser, get the predicted KD instantly — no Python needed.
- **Exact reproduction package** (`reproduce/`): frozen environment + self-checking script that rebuilds the published model bit-for-bit from the curated parameter table.
- **Bug fixes in the prediction utilities** (see *v1 → v2 fixes* below): v1's `INPUT_FEATURES` accidentally included the target column and dropped `S_geometry`, and the legacy `calculate_geometry_score` used an integer threshold version that did not match the training data.

---

## Quick start — three ways to use ABLE

### Option A: Web app (no installation)
Open the browser predictor and fill in the 9 raw measurements for your construct:

> **https://jinhua9.github.io/ABLE-nanobody/**

Everything runs locally in your browser (ONNX inference); no data leaves your machine.

### Option B: Python with the pre-trained model
```bash
git clone https://github.com/Jinhua9/ABLE-nanobody.git
cd ABLE-nanobody
pip install -r requirements.txt
```

```python
import pandas as pd
from src.able_model import load_able_model
from src.utils import prepare_model_input

model = load_able_model('models/ABLE_model.pkl')

# Raw measurements — KDs in nM; derived features are computed for you
new_data = pd.DataFrame({
    'KD_N': [2.5], 'KD_C': [3.0],        # monomer KDs (nM)
    'D_mono': [40.2], 'd_epi': [15.3], 'L_link': [45.0],
    'AN': [450.2], 'AC': [420.8],
    'n_bonds_N': [12], 'n_bonds_C': [10]})

X = prepare_model_input(new_data)   # derives ratios, logs KDs, orders columns
pred_KD = model.predict(X)          # returns KD in nM
print(f"Predicted KD: {pred_KD[0]:.3f} nM")
```

`prepare_model_input` computes `R_link = L_link/D_mono`, `R_area = A_N/A_C`, `R_bond = n_t^C/n_t^N` and the continuous geometry score `S_geometry`, then log-transforms the KD columns and returns the 13 columns in the exact order the model expects. (If you already supply the 13 features yourself, remember that `KD_N`/`KD_C` must be in **ln scale**.)

### Option C: Run the examples
```bash
python examples/01_train_model.py   # train a fresh model on the 102-construct dataset
python examples/02_predict_avidity.py  # predict avidity for example constructs
```
Expected run time: < 10 seconds each on a normal desktop.

---

## System Requirements

- Linux (Ubuntu 20.04+), macOS (11.0+), or Windows 10/11 (native Python or WSL)
- Python 3.9 – 3.12
- Standard desktop/laptop; no GPU required; < 100 MB disk

---

## Input features

All features are computed from AlphaFold-predicted antigen–nanobody complexes (calculation protocols in the paper's Methods and Supplementary Information).

| # | Feature | Definition |
|---|---------|------------|
| 1 | `KD_N` | K_D of the N-terminal monomer (enters the model as ln, in nM) |
| 2 | `KD_C` | K_D of the C-terminal monomer (enters the model as ln, in nM) |
| 3 | `D_mono` | distance between the two nanobody C-termini (Å) |
| 4 | `d_epi` | epitope distance (Å, negative on epitope clash) |
| 5 | `L_link` | linker contour length (Å) |
| 6 | `R_link` | `L_link / D_mono` |
| 7 | `R_area` | `A_N / A_C` (buried SASA per terminus, Å²) |
| 8 | `R_bond` | `n_t^C / n_t^N` (interface contacts per terminus) |
| 9 | `AN` | N-terminal buried SASA (Å²) |
| 10 | `AC` | C-terminal buried SASA (Å²) |
| 11 | `n_bonds_N` | N-terminal interface contacts |
| 12 | `n_bonds_C` | C-terminal interface contacts |
| 13 | `S_geometry` | continuous geometry score, 0–3 (see below) |

Geometry score (as used to build the training data):

```
S_geometry = (1 - min(|R_link - 1|, 1)) + (1 - min(|R_area - 1|, 0.5)) + (1 - min(|R_bond - 1|, 0.5))
```

---

## Reproducing the published results

The published model's exact training protocol lives in [`reproduce/`](reproduce/):

```bash
cd reproduce
pip install -r requirements.txt      # frozen versions
python reproduce_able_gb.py
```

The script rebuilds the model from `reproduce/data/Parameters_3200s.xlsx` (the curated parameter table used in the paper) with the published split (stratified on ln(KD) deciles, `random_state=123`) and hyperparameters (`random_state=2`), and finishes with a built-in self-check:

```text
Train R² = 0.9525   Test R² = 0.8707   R² diff = 0.0819
Early-stopped trees: 54/150
OK REPRODUCED exactly (tol 1e-06)
```

`examples/01_train_model.py` is a lighter demonstration that trains on `data/ABLE Dataset.xlsx` with a demonstration split (seed 456); with 21 test samples, single-split test R² varies strongly across seeds, so use `reproduce/` for the published numbers.

### v1 → v2 fixes (for transparency)
- `INPUT_FEATURES` in v1 was defined as `FEATURE_NAMES[:-1]`, which was intended to exclude the target `KD_bipara` but actually kept `KD_bipara` (position 3) and dropped `S_geometry` (last). v2 lists the 13 input features explicitly.
- v1's `calculate_geometry_score` used an integer 0–3 threshold rule that does not match the `S_geometry` column of the training data; v2 implements the continuous capped-penalty formula above (verified to match the training table exactly, 102/102 rows).
- v1 examples passed raw nM KDs and a misordered feature matrix to `predict()`; v2's `prepare_model_input` fixes both.

---

## Repository structure

```
ABLE-nanobody/
├── models/
│   ├── ABLE_model.pkl            # v2 published model (13 ln-scale features)
│   └── ABLE_model_v1_legacy.pkl  # original v1 model (kept for reference)
├── src/
│   ├── able_model.py             # ABLEModel class + geometry score
│   └── utils.py                  # prepare_model_input, validation, reports
├── examples/
│   ├── 01_train_model.py         # train a fresh model on the 102-construct dataset
│   └── 02_predict_avidity.py     # predict avidity for new constructs
├── docs/                        # browser predictor (GitHub Pages)
│   ├── index.html
│   └── able_model.onnx           # ONNX export, numerically identical to the pkl
├── reproduce/                    # exact reproduction of the published model
│   ├── reproduce_able_gb.py      # self-checking script
│   ├── requirements.txt          # frozen versions
│   └── data/Parameters_3200s.xlsx
├── data/
│   └── ABLE Dataset.xlsx         # 102-construct dataset (raw measurements)
└── requirements.txt
```

---

## License

This project is licensed under the MIT License – see the LICENSE file for details.

---

## Contact & Support

Corresponding authors: Jiahai Shi (jh.shi@nus.edu.sg)

Repository maintainer: Jinhua Gong (jinhgong-c@my.cityu.edu.hk)

Issue tracker: GitHub Issues

---

## Acknowledgements
This work was supported by the Ministry of Education (MOE) of Singapore and Shenzhen BGI Research.
