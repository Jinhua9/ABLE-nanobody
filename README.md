# ABLE: Biparatopic Nanobody Avidity Prediction

## Overview
This repository contains the implementation of the ABLE model, a gradient boosting regression model for predicting biparatopic nanobody avidity from structural and energetic features, as described in:

**"High-avidity biparatopic nanobodies designed by ABLE"**

ABLE enables the rational design of high-avidity biparatopic nanobodies by predicting apparent dissociation constants (KD) from 14 structural features derived from AlphaFold-predicted complexes.

## Key Features
- **Accurate Prediction**: R² = 0.84 on test set, 90.50% 2-fold accuracy
- **Interpretable Features**: 14 physically meaningful features, including epitope distance, contact area ratios, and linker geometry
- **Fast Training**: <1 second training time for 102 samples
- **Easy Integration**: Simple API for training and prediction

## Installation

```bash
# Clone the repository
git clone https://github.com/Jinhua9/ABLE-nanobody.git
cd ABLE-nanobody

# Install dependencies
pip install -r requirements.txt


### 2. `requirements.txt`

```txt
# Core dependencies
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
scipy==1.10.1
joblib==1.2.0

# Optional for advanced users
matplotlib==3.7.1  # for visualization
seaborn==0.12.2    # for visualization
jupyter==1.0.0     # for notebooks


Quick Start
1. Training the Model

from src.able_model import ABLEModel
import pandas as pd

# Load dataset
df = pd.read_csv('data/features_integrated.csv')

# Train model
model = ABLEModel()
model.train_from_dataframe(df, target_col='KD_bipara_nM')

# Save model
model.save('models/able_model.pkl')

2. Making Predictions
from src.able_model import load_able_model

# Load trained model
model = load_able_model('models/able_model.pkl')

# Predict avidity for new constructs
predictions = model.predict(features_df)

Dataset Structure
The dataset (data/features_integrated.csv) contains 102 biparatopic nanobody constructs with 14 features:

Feature	Description	Unit
KD_N	N-terminal monomer dissociation constant	nM
KD_C	C-terminal monomer dissociation constant	nM
D_mono	Distance between monomer termini	Å
d_epi	Minimum epitope distance (negative for steric clash)	Å
L_link	Linker contour length	Å
R_link	Linker adequacy ratio (L_link/D_mono)	-
R_area	Contact area ratio (AN/AC)	-
R_bond	Bond number ratio (n_bonds_C/n_bonds_N)	-
AN	Buried SASA for N-terminal monomer	Å²
AC	Buried SASA for C-terminal monomer	Å²
n_bonds_N	Interfacial bonds for N-terminal monomer	count
n_bonds_C	Interfacial bonds for C-terminal monomer	count
S_geometry	Composite geometry score (0-3)	-
KD_bipara	Biparatopic construct dissociation constant (target)	nM

Citation
If you use ABLE in your research, please cite:
@article{gong2024able,
  title={AlphaFold-assisted high-avidity biparatopic nanobodies designed by ABLE},
  author={Gong, Jinhua and Zhu, Liang and Wei, Likun and Wang, Meiniang and Deng, Xin and Tang, Zimin and Wang, Wanyu and Li, Zhiyong and Yu, Siyuan and Pan, Haifeng and Zhuang, Haoyun and Tan, Shanzhi and Yu, Xuan and Guo, Zhixuan and Liu, Dasheng and Zheng, Penglong and Ge, Shengxiang and Xia, Ningshao and Shi, Jiahai},
  journal={Nature Methods},
  year={2024}
}

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For questions about the model or dataset, please open an issue or contact Jinhua GONG.

