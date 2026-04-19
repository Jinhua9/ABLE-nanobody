"""
Utility functions for ABLE model
"""

import numpy as np
import pandas as pd

def validate_features(df, required_features=None):
    """
    Validate that the DataFrame contains all required features.
    
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
    
    print(f"✓ All required features present")
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
        print("✓ Calculated R_link = L_link / D_mono")
    
    # Calculate R_area if not present
    if 'R_area' not in df.columns and 'AN' in df.columns and 'AC' in df.columns:
        df['R_area'] = df['AN'] / df['AC']
        print("✓ Calculated R_area = AN / AC")
    
    # Calculate R_bond if not present
    if 'R_bond' not in df.columns and 'n_bonds_C' in df.columns and 'n_bonds_N' in df.columns:
        df['R_bond'] = df['n_bonds_C'] / df['n_bonds_N']
        print("✓ Calculated R_bond = n_bonds_C / n_bonds_N")
    
    # Calculate S_geometry if not present
    if 'S_geometry' not in df.columns:
        from .able_model import calculate_geometry_score
        
        # Ensure required ratios exist
        if 'R_link' not in df.columns or 'R_area' not in df.columns or 'R_bond' not in df.columns:
            print("⚠ Cannot calculate S_geometry: missing ratio columns")
        else:
            df['S_geometry'] = df.apply(
                lambda row: calculate_geometry_score(
                    row['R_link'],
                    row['R_area'],
                    row['R_bond']
                ), axis=1
            )
            print("✓ Calculated S_geometry (0-3 scale)")
    
    return df


def print_model_summary(model):
    """
    Print the summary of the trained ABLE model.
    
    Args:
        model (ABLEModel): Trained model instance
    """
    print("=" * 60)
    print("ABLE MODEL SUMMARY")
    print("=" * 60)
    
    print(f"Model Type: Gradient Boosting Regressor")
    print(f"Number of Features: {len(model.feature_names)}")
    print(f"Trained: {model.is_trained}")
    
    if model.is_trained:
        print(f"Number of Trees Trained: {getattr(model, 'actual_trees_trained', 'N/A')}")
        
        # Feature importance
        try:
            importance_df = model.get_feature_importance()
            print("\nFEATURE IMPORTANCE (Top 6):")
            print("-" * 40)
            for i, row in importance_df.head(6).iterrows():
                print(f"  {i+1:2d}. {row['feature']:15} {row['importance']:.4f}")
            
            # Show cumulative importance
            cum_90 = importance_df[importance_df['cumulative'] <= 0.9]
            print(f"\nFeatures explaining 90% of variance: {len(cum_90)}")
        except:
            print("\nFeature importance not available")
    
    print("=" * 60)


def check_data_quality(df):
    """
    Check data quality and report issues.
    
    Args:
        df (pd.DataFrame): Input data
        
    Returns:
        dict: Quality report
    """
    report = {}
    
    # Check for missing values
    missing = df.isnull().sum()
    report['missing_values'] = missing[missing > 0].to_dict()
    
    # Check for infinite values
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    report['infinite_values'] = inf_count
    
    # Check feature ranges
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    ranges = {}
    for col in numeric_cols:
        ranges[col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'std': df[col].std()
        }
    report['feature_ranges'] = ranges
    
    # Check for negative KD values
    kd_cols = [col for col in df.columns if 'KD' in col]
    negative_kd = {}
    for col in kd_cols:
        neg_count = (df[col] <= 0).sum()
        if neg_count > 0:
            negative_kd[col] = neg_count
    report['negative_kd'] = negative_kd
    
    return report


def create_prediction_report(y_true, y_pred):
    """
    Create comprehensive prediction report.
    
    Args:
        y_true (array): True KD values (nM)
        y_pred (array): Predicted KD values (nM)
        
    Returns:
        dict: Prediction metrics
    """
    # Ensure numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Convert to log scale for some metrics
    y_true_log = np.log(y_true)
    y_pred_log = np.log(y_pred)
    
    # Calculate basic metrics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    metrics = {
        'r2': r2_score(y_true_log, y_pred_log),
        'rmse_log': np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
        'mae_log': mean_absolute_error(y_true_log, y_pred_log),
        'rmse_nM': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae_nM': mean_absolute_error(y_true, y_pred),
    }
    
    # Calculate fold accuracy
    ratios = np.maximum(y_pred / y_true, y_true / y_pred)
    metrics['accuracy_2fold'] = np.mean(ratios <= 2) * 100
    metrics['accuracy_5fold'] = np.mean(ratios <= 5) * 100
    
    # Calculate relative errors
    relative_errors = np.abs(y_pred - y_true) / y_true
    metrics['median_relative_error'] = np.median(relative_errors)
    metrics['mean_relative_error'] = np.mean(relative_errors)
    metrics['std_relative_error'] = np.std(relative_errors)
    
    # Count predictions in different error ranges
    error_ranges = {
        'within_10%': np.sum(relative_errors <= 0.1) / len(y_true) * 100,
        'within_20%': np.sum(relative_errors <= 0.2) / len(y_true) * 100,
        'within_50%': np.sum(relative_errors <= 0.5) / len(y_true) * 100,
        'within_100%': np.sum(relative_errors <= 1.0) / len(y_true) * 100,
    }
    metrics['error_ranges'] = error_ranges
    
    return metrics
