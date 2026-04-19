# ABLE: AlphaFold-assisted Biparatopic Nanobody Avidity Prediction

## Overview
This repository contains the implementation of the ABLE model, a gradient boosting regression model for predicting biparatopic nanobody avidity from structural and energetic features, as described in:

**"AlphaFold-assisted high-avidity biparatopic nanobodies designed by ABLE"**  

ABLE enables rational design of high-avidity biparatopic nanobodies by predicting apparent dissociation constants (KD) based on 14 structural features derived from AlphaFold-predicted complexes.

---

## System Requirements

### Operating Systems
- Linux (Ubuntu 20.04 or later)
- macOS (11.0 or later)
- Windows 10/11 (via WSL or native Python)

### Software Dependencies
- Python 3.8 – 3.11
- Package versions as listed in `requirements.txt`

### Hardware Requirements
- Standard desktop or laptop (no GPU required)
- RAM: ≥ 4 GB
- Disk space: < 100 MB

### Tested Configurations
- Ubuntu 22.04, Python 3.9.18
- macOS Ventura 13.6, Python 3.10.13
- Windows 11 (WSL2), Python 3.10.11

---

## Installation Guide

**Typical install time: < 2 minutes** on a normal desktop

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Jinhua9/ABLE-nanobody.git
   cd ABLE-nanobody
   
2. **Create a virtual environment (recommended)**  
   ```bash
   python -m venv able_env
   source able_env/bin/activate   # Linux/macOS
   or .\able_env\Scripts\activate   # Windows
   
3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   
4. **Verify installation**  
   ```bash
   python -c "import sklearn, pandas, joblib; print('OK')"

---

## Demo
We provide a complete demo that trains the ABLE model on the full dataset (102 biparatopic constructs) and evaluates its performance.

1. **Run the demo**  
   ```bash
   python examples/01_train_model.py
   
2. **Expected output (abbreviated)**  
   ```text
    ABLE MODEL TRAINING EXAMPLE
   Training on the complete dataset (102 biparatopic constructs)
   ...
   PERFORMANCE ON TEST SET (20% of data)
   R² (log scale):            0.84
   2-fold Accuracy:           90.5%
   ...
   Top 10 Most Important Features:
   1. d_epi            0.2145
   2. KD_C             0.1783
   ...
3. **Expected run time: < 10 seconds on a normal desktop**  
   The demo uses the full dataset and an 80/20 stratified train-test split (random seed 456). Results may vary slightly but should approximate the R² ~0.84 and 2-fold accuracy ~90% reported in the paper.

---

## Instructions for Use

1. **Predict avidity for your own biparatopic constructs**  
You need to compute the 13 input features (all except KD_bipara) for your constructs. Then use the pre-trained model:
   ```python
   from src.able_model import load_able_model
   import pandas as pd

   # Load pre-trained model
   model = load_able_model('models/ABLE_model.pkl')

   # Prepare your feature DataFrame (must have exactly these 13 columns)
   new_data = pd.DataFrame({
    'KD_N': [2.5], 'KD_C': [3.0], 'D_mono': [40.2], 'd_epi': [15.3],
    'L_link': [45.0], 'R_link': [1.12], 'R_area': [0.95], 'R_bond': [1.05],
    'AN': [450.2], 'AC': [420.8], 'n_bonds_N': [12], 'n_bonds_C': [10],
    'S_geometry': [3]})

   # Predict (model returns KD in nM)
   pred_KD = model.predict(new_data.values)
   print(f"Predicted KD: {pred_KD[0]:.3f} nM")
  
2. **Feature definitions**  
   All features must be computed from AlphaFold-predicted antigen–nanobody complexes. Detailed calculation protocols are provided in the Methods and Supplementary Information sections of the paper.
    
3. **Train your own model**  
   ```bash
   python examples/01_train_model.py

4. **Batch prediction**  
   ```python
   import pandas as pd
   from src.able_model import load_able_model
   model = load_able_model('models/ABLE_model.pkl')
   new_data = pd.read_excel('your_constructs.xlsx')  # must contain the 13 input columns
   predictions = model.predict(new_data.values)
   new_data['Predicted_KD_nM'] = predictions
   new_data.to_excel('predictions.xlsx', index=False)

---

## Reproducing paper results

1. **Notes**  
All quantitative results reported in the manuscript (R², fold accuracy, feature importance, etc.) can be reproduced by running:
   ```python
   python examples/01_train_model.py
The script uses the exact same dataset (data/ABLE Dataset.xlsx) and the same random seed (456) as described in the paper.

---

## License

This project is licensed under the MIT License – see the LICENSE file for details.

---

##  Contact & Support

Corresponding authors: Jiahai Shi (jh.shi@nus.edu.sg)

Repository maintainer: Jinhua Gong (jinhgong-c@my.cityu.edu.hk)

Issue tracker: GitHub Issues

---

## Acknowledgements
This work was supported by the Ministry of Education (MOE) of Singapore and Shenzhen BGI Research.
