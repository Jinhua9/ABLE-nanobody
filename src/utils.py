"""
Utility functions for ABLE model
"""

import numpy as np
import pandas as pd

def validate_features(df, required_features=None):
    """
    Validate that DataFrame contains all required features.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        required_features (list): List of required feature names
        
    Returns:
        bool: True if all features are present
    """
    if required_features is None:
        required_features = [
            'KD_N', 'KD_C', 'D_mono', 'd_epi', 'L_link',
            'R_link', 'R_area', 'R_bond', 'AN', 'AC',
            'n_bonds_N', 'n_bonds_C', 'S_geometry'
        ]
    
    missing = [f for f in required_features if f not in df.columns]
    
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    
    return True


def calculate_derived_features(df):
    """
    Calculate derived features from basic measurements.
    
    Args:
        df (pd.DataFrame): Input DataFrame with basic features
        
    Returns:
        pd.DataFrame: DataFrame with additional derived features
    """
    df = df.copy()
    
    # Calculate R_link if not present
    if 'R_link' not in df.columns and 'L_link' in df.columns and 'D_mono' in df.columns:
        df['R_link'] = df['L_link'] / df['D_mono']
    
    # Calculate R_area if not present
    if 'R_area' not in df.columns and 'AN' in df.columns and 'AC' in df.columns:
        df['R_area'] = df['AN'] / df['AC']
    
    # Calculate R_bond if not present
    if 'R_bond' not in df.columns and 'n_bonds_C' in df.columns and 'n_bonds_N' in df.columns:
        df['R_bond'] = df['n_bonds_C'] / df['n_bonds_N']
    
    # Calculate S_geometry if not present
    if 'S_geometry' not in df.columns:
        from src.able_model import create_geometry_score
        df['S_geometry'] = df.apply(
            lambda row: create_geometry_score(
                row.get('R_link', 1.0),
                row.get('R_area', 1.0),
                row.get('R_bond', 1.0)
            ), axis=1
        )
    
    return df


def print_model_summary(model):
    """
    Print summary of trained ABLE model.
    
    Args:
        model (ABLEModel): Trained model instance
    """
    print("=" * 60)
    print("ABLE Model Summary")
    print("=" * 60)
    
    print(f"Model Type: Gradient Boosting Regressor")
    print(f"Number of Features: {len(model.feature_names)}")
    print(f"Trained: {model.is_trained}")
    
    if model.is_trained:
        print(f"Number of Trees Trained: {model.actual_trees_trained}")
        
        # Feature importance
        importance_df = model.get_feature_importance()
        print("\nTop 5 Most Important Features:")
        for i, row in importance_df.head().iterrows():
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    print("=" * 60)
