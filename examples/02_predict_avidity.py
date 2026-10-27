"""
Example 2: Use trained ABLE model to predict avidity for new constructs

The v2 model expects ln(KD) features and a fixed column order.
``utils.prepare_model_input`` handles all of that from raw nM
measurements, so this example reads exactly like the v1 workflow.
"""

import os
import sys
import pandas as pd

# Make 'src' importable and paths robust regardless of the working
# directory the example is launched from
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
os.chdir(_HERE)  # example data paths below are relative to this folder

from src.able_model import load_able_model
from src.utils import prepare_model_input

def main():
    print("Predicting Biparatopic Avidity with ABLE")
    print("=" * 60)

    # Load trained model
    print("Loading trained ABLE model...")
    model = load_able_model('../models/ABLE_model.pkl')

    # Create example data for prediction (raw measurements, KD in nM)
    print("\nCreating example data...")
    example_data = pd.DataFrame({
        'KD_N': [2.5, 5.0, 1.0],          # N-terminal monomer KD (nM)
        'KD_C': [3.0, 6.0, 2.0],          # C-terminal monomer KD (nM)
        'D_mono': [40.2, 35.8, 45.1],     # Distance between termini (Å)
        'd_epi': [15.3, -5.2, 25.1],      # Epitope distance (Å, negative for clash)
        'L_link': [45.0, 30.0, 60.0],     # Linker contour length (Å)
        'AN': [450.2, 380.5, 520.3],      # Buried SASA for N-terminal (Å²)
        'AC': [420.8, 400.2, 480.6],      # Buried SASA for C-terminal (Å²)
        'n_bonds_N': [12, 8, 15],         # Interfacial bonds for N-terminal
        'n_bonds_C': [10, 9, 14],         # Interfacial bonds for C-terminal
    })

    # Build the model-ready feature matrix:
    #   * derives R_link / R_area / R_bond / S_geometry
    #   * log-transforms KD_N / KD_C (nM -> ln)
    #   * orders the 13 columns exactly as the model expects
    X = prepare_model_input(example_data)
    print(f"Model input shape: {X.shape}")
    print(f"\nModel-ready features:")
    print(X)

    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X)

    # Display results
    print("\nPrediction Results:")
    print("=" * 40)
    for i, (features, pred) in enumerate(zip(example_data.to_dict('records'), predictions)):
        print(f"\nConstruct {i+1}:")
        print(f"  Features:")
        print(f"    KD_N: {features['KD_N']:.1f} nM")
        print(f"    KD_C: {features['KD_C']:.1f} nM")
        print(f"    d_epi: {features['d_epi']:.1f} Å")
        print(f"    R_link: {features['L_link'] / features['D_mono']:.2f}")
        print(f"    S_geometry: {X.iloc[i]['S_geometry']:.2f}")
        print(f"  Predicted KD: {pred:.3f} nM")

        # Calculate avidity gain
        max_monomer_kd = max(features['KD_N'], features['KD_C'])
        gain = max_monomer_kd / pred
        print(f"  Avidity gain: {gain:.1f}-fold")

    print("\n" + "=" * 60)
    print("Prediction complete!")

if __name__ == "__main__":
    main()
