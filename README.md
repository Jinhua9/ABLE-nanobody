# ABLE: AlphaFold-assisted Biparatopic Nanobody Avidity Prediction

## Overview
This repository contains the implementation of the ABLE model, a gradient boosting regression model for predicting biparatopic nanobody avidity from structural and energetic features, as described in:

**"AlphaFold-assisted high-avidity biparatopic nanobodies designed by ABLE"**

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

### Install dependencies
pip install -r requirements.txt

## Quick Start
### Option 1: Load Pre-trained Model
from src.able_model import load_able_model

1. Load the pre-trained model
model = load_able_model('models/ABLE_model.pkl')

2. Prepare your features (see dataset format)
features = [...]  # Your 14 features in order

3. Predict KD
predicted_kd = model.predict([features])
print(f"Predicted KD: {predicted_kd[0]:.3f} nM")

### Option 2: Train Model from Scratch
1. Run the training example
python examples/01_train_model.py

## Usage Examples
### Example 1: Training the Model
from src.able_model import ABLEModel
import pandas as pd

1. Load dataset
df = pd.read_excel('data/ABLE Dataset.xlsx')

2. Initialize and train model
model = ABLEModel()
X_test, y_test, y_pred = model.train_from_excel('data/ABLE Dataset.xlsx')

3. Evaluate performance
metrics = model.evaluate(X_test, y_test)
print(f"Test R²: {metrics['r2']:.4f}")
print(f"2-fold Accuracy: {metrics['accuracy_2fold']:.1f}%")

### Example 2: Batch Prediction
import pandas as pd
from src.able_model import load_able_model

1. Load model and new data
model = load_able_model('models/ABLE_model.pkl')
new_data = pd.read_excel('new_constructs.xlsx')

3. Predict
predictions = model.predict(new_data.values)
new_data['Predicted_KD_nM'] = predictions
new_data.to_excel('predictions.xlsx', index=False)

## Contact
For questions about the model or dataset, please open an issue or contact Jinhua GONG and Jiahai Shi.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
