# ABLE:Biparatopic Nanobody Avidity Prediction

## Overview
This repository contains the implementation of the ABLE model, a gradient boosting regression model for predicting biparatopic nanobody avidity from structural and energetic features, as described in:

**"High-avidity biparatopic nanobodies designed by ABLE"**

ABLE enables rational design of high-avidity biparatopic nanobodies by predicting apparent dissociation constants (KD) based on 14 structural features derived from AlphaFold-predicted complexes.

## Repository Structure
ABLE-nanobody/
├── data/
│ └── ABLE Dataset.xlsx # Complete dataset of 102 biparatopic constructs
├── models/
│ └── ABLE_model.pkl # Pre-trained ABLE model
├── src/
│ ├── init.py # Package initialization
│ ├── able_model.py # ABLE model implementation
│ └── utils.py # Utility functions
├── examples/
│ ├── 01_train_model.py # Example: Train model from scratch
│ └── 02_predict_avidity.py # Example: Make predictions with trained model
├── requirements.txt # Python dependencies
├── LICENSE # MIT License
└── README.md # This file

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/Jinhua9/ABLE-nanobody.git
cd ABLE-nanobody

# Install dependencies
pip install -r requirements.txt
