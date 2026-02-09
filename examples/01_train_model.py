"""
Example 1: Train ABLE model on the complete dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.able_model import ABLEModel
from src.utils import print_model_summary, create_prediction_report

def main():
    print("=" * 70)
    print("ABLE MODEL TRAINING EXAMPLE")
    print("Training on complete dataset (102 biparatopic constructs)")
    print("=" * 70)
    
    try:
        # Load dataset
        print("\n1. Loading dataset...")
        data_path = os.path.join('..', 'data', 'ABLE Dataset.xlsx')
        df = pd.read_excel(data_path)
        print(f"   ✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
        
        # Initialize model
        print("\n2. Initializing ABLE model...")
        model = ABLEModel()
        print("   ✓ Model initialized with optimal hyperparameters")
        
        # Train model (80/20 split)
        print("\n3. Training model (80% train, 20% test)...")
        X_test, y_test, y_pred = model.train_from_excel(
            data_path,
            test_size=0.2,
            random_state=456
        )
        
        # Evaluate model
        print("\n4. Evaluating model performance...")
        metrics = model.evaluate(X_test, y_test)
        
        print("\n" + "=" * 50)
        print("PERFORMANCE ON TEST SET (20% of data)")
        print("=" * 50)
        print(f"R² (log scale):            {metrics['r2']:.4f}")
        print(f"RMSE (log scale):          {metrics['rmse']:.4f}")
        print(f"MAE (log scale):           {metrics['mae']:.4f}")
        print(f"RMSE (nM scale):           {metrics['rmse_nM']:.2f} nM")
        print(f"MAE (nM scale):            {metrics['mae_nM']:.2f} nM")
        print(f"2-fold Accuracy:           {metrics['accuracy_2fold']:.1f}%")
        print(f"5-fold Accuracy:           {metrics['accuracy_5fold']:.1f}%")
        print(f"Median Relative Error:     {metrics['median_relative_error']:.3f}")
        print(f"Mean Relative Error:       {metrics['mean_relative_error']:.3f}")
        
        # Feature importance
        print("\n5. Feature Importance Analysis...")
        importance_df = model.get_feature_importance()
        print("\nTop 10 Most Important Features:")
        print("-" * 40)
        for i, row in importance_df.head(10).iterrows():
            print(f"{i+1:2d}. {row['feature']:15} {row['importance']:.4f}")
        
        # Save model
        print("\n6. Saving trained model...")
        model_path = os.path.join('..', 'models', 'trained_model.pkl')
        model.save(model_path)
        print(f"   ✓ Model saved to: {model_path}")
        
        # Model summary
        print("\n7. Model Summary...")
        print_model_summary(model)
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: File not found - {e}")
        print("Please ensure the data file exists at: data/ABLE Dataset.xlsx")
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
