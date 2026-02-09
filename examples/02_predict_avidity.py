"""
Example 2: Use trained ABLE model to predict avidity for new constructs
"""

import pandas as pd
import numpy as np
from src.able_model import load_able_model
from src.utils import calculate_derived_features

def main():
    print("Predicting Biparatopic Avidity with ABLE")
    print("=" * 60)
    
    # Load trained model
    print("Loading trained ABLE model...")
    model = load_able_model('../models/able_model.pkl')
    
    # Create example data for prediction
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
    
    # Calculate derived features
    example_data = calculate_derived_features(example_data)
    print(f"Example data shape: {example_data.shape}")
    print(f"\nExample features:")
    print(example_data)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(example_data.values)
    
    # Display results
    print("\nPrediction Results:")
    print("=" * 40)
    for i, (features, pred) in enumerate(zip(example_data.to_dict('records'), predictions)):
        print(f"\nConstruct {i+1}:")
        print(f"  Features:")
        print(f"    KD_N: {features['KD_N']:.1f} nM")
        print(f"    KD_C: {features['KD_C']:.1f} nM")
        print(f"    d_epi: {features['d_epi']:.1f} Å")
        print(f"    R_link: {features.get('R_link', 'N/A'):.2f}")
        print(f"    S_geometry: {features.get('S_geometry', 'N/A')}")
        print(f"  Predicted KD: {pred:.3f} nM")
        
        # Calculate avidity gain
        max_monomer_kd = max(features['KD_N'], features['KD_C'])
        gain = max_monomer_kd / pred
        print(f"  Avidity gain: {gain:.1f}-fold")
    
    print("\n" + "=" * 60)
    print("Prediction complete!")

if __name__ == "__main__":
    main()
