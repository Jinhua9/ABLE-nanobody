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
