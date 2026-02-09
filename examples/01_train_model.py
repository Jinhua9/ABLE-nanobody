"""
Example 1: Train ABLE model on the integrated dataset
"""

import pandas as pd
import numpy as np
from src.able_model import ABLEModel
from src.utils import print_model_summary

def main():
    print("Training ABLE Model")
    print("=" * 60)
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('../data/features_integrated.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    
    # Initialize model
    print("\nInitializing ABLE model...")
    model = ABLEModel()
    
    # Train model
    print("Training model...")
    X_test, y_test, y_pred = model.train_from_dataframe(
        df, 
        target_col='KD_bipara_nM',
        test_size=0.2,
        random_state=456
    )
    
    # Evaluate model
    print("\nEvaluating model performance...")
    metrics = model.evaluate(X_test, y_test)
    
    print("\nPerformance Metrics:")
    print(f"  Test R²: {metrics['r2']:.4f}")
    print(f"  Test RMSE: {metrics['rmse']:.4f} (log scale)")
    print(f"  Test MAE: {metrics['mae']:.4f} (log scale)")
    print(f"  2-fold Accuracy: {metrics['accuracy_2fold']:.1f}%")
    print(f"  5-fold Accuracy: {metrics['accuracy_5fold']:.1f}%")
    print(f"  Median Relative Error: {metrics['median_relative_error']:.4f}")
    
    # Print model summary
    print("\n" + "=" * 60)
    print_model_summary(model)
    
    # Save model
    print("\nSaving model...")
    model.save('../models/able_model.pkl')
    print("Model saved to '../models/able_model.pkl'")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()
