"""
ABLE Model Implementation
AlphaFold-assisted Biparatopic Nanobody Avidity Prediction
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class ABLEModel:
    """
    ABLE: AlphaFold-assisted Biparatopic Nanobody Avidity Prediction Model
    
    This model predicts the apparent dissociation constant (KD) of biparatopic
    nanobody constructs using 14 structural and energetic features derived
    from AlphaFold-predicted complexes.
    """
    
    # Required features for prediction
    FEATURE_NAMES = [
        'KD_N', 'KD_C', 'KD_bipara', 'D_mono', 'd_epi', 'L_link',
        'R_link', 'R_area', 'R_bond', 'AN', 'AC', 'n_bonds_N',
        'n_bonds_C', 'S_geometry'
    ]
    
    # Target column (natural log of KD in nM)
    TARGET_COL = 'KD_bipara'
    
    # Optimal hyperparameters determined by randomized search
    OPTIMAL_PARAMS = {
        'n_estimators': 100,
        'learning_rate': 0.2,
        'loss': 'absolute_error',
        'criterion': 'friedman_mse',
        'min_samples_split': 4,
        'min_samples_leaf': 2,
        'min_weight_fraction_leaf': 0.0,
        'subsample': 0.95,
        'max_depth': 12,
        'max_features': 0.7,
        'min_impurity_decrease': 0.01,
        'ccp_alpha': 0.01,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10,
        'tol': 0.0001,
        'random_state': 3,
        'alpha': 0.9,
        'verbose': 0,
        'warm_start': False
    }
    
    def __init__(self, params=None):
        """
        Initialize ABLE model.
        
        Args:
            params (dict, optional): Model hyperparameters. If None, uses optimal parameters.
        """
        if params is None:
            params = self.OPTIMAL_PARAMS
        
        self.params = params
        self.model = GradientBoostingRegressor(**params)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = self.FEATURE_NAMES[:-1]  # Exclude target
        
    def prepare_data(self, df, target_col=None):
        """
        Prepare features and target from DataFrame.
        
        Args:
            df (pd.DataFrame): Input data containing features and target
            target_col (str, optional): Target column name. Defaults to 'KD_bipara'
            
        Returns:
            tuple: (X_features, y_target) as numpy arrays
        """
        if target_col is None:
            target_col = self.TARGET_COL
        
        # Check required features
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Extract features
        X = df[self.feature_names].values
        
        # Extract target (natural log of KD in nM)
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        y = df[target_col].values
        
        # Convert to natural log if not already
        if np.max(y) > 100:  # Assuming nM scale
            y = np.log(y)
        
        return X, y
    
    def train(self, X_train, y_train):
        """
        Train the ABLE model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target (ln(KD))
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Store actual number of trees trained (may differ due to early stopping)
        self.actual_trees_trained = self.model.n_estimators_
    
    def train_from_dataframe(self, df, target_col=None, test_size=0.2, random_state=456):
        """
        Train model directly from DataFrame with train/test split.
        
        Args:
            df (pd.DataFrame): Input data
            target_col (str, optional): Target column name
            test_size (float): Proportion for test set
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_test, y_test, y_pred) for evaluation
        """
        # Prepare data
        X, y = self.prepare_data(df, target_col)
        
        # Split data (stratified by target quantiles for balanced distribution)
        from sklearn.model_selection import StratifiedShuffleSplit
        from sklearn.preprocessing import KBinsDiscretizer
        
        # Create bins for stratified split
        discretizer = KBinsDiscretizer(n_bins=min(10, len(y)//10), 
                                      encode='ordinal', 
                                      strategy='quantile')
        y_binned = discretizer.fit_transform(y.reshape(-1, 1)).flatten().astype(int)
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, 
                                     random_state=random_state)
        
        for train_idx, test_idx in sss.split(X, y_binned):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        self.train(X_train, y_train)
        
        # Make predictions on test set
        y_pred = self.predict(X_test)
        
        return X_test, y_test, y_pred
    
    def predict(self, X):
        """
        Predict biparatopic dissociation constants.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Predicted KD values in nM (back-transformed from log scale)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict (model outputs ln(KD))
        y_pred_log = self.model.predict(X_scaled)
        
        # Convert back to nM scale
        y_pred = np.exp(y_pred_log)
        
        return y_pred
    
    def evaluate(self, X_test, y_test_true):
        """
        Evaluate model performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test_true (np.ndarray): True KD values in nM
            
        Returns:
            dict: Evaluation metrics
        """
        # Predict
        y_pred = self.predict(X_test)
        
        # Convert y_test to log scale for comparison
        y_test_log = np.log(y_test_true)
        y_pred_log = np.log(y_pred)
        
        # Calculate metrics
        metrics = {
            'r2': r2_score(y_test_log, y_pred_log),
            'rmse': np.sqrt(mean_squared_error(y_test_log, y_pred_log)),
            'mae': mean_absolute_error(y_test_log, y_pred_log),
            'r2_original': r2_score(y_test_true, y_pred),
            'rmse_original': np.sqrt(mean_squared_error(y_test_true, y_pred)),
            'mae_original': mean_absolute_error(y_test_true, y_pred),
        }
        
        # Calculate fold accuracy
        ratios = np.maximum(y_pred / y_test_true, y_test_true / y_pred)
        metrics['accuracy_2fold'] = np.mean(ratios <= 2) * 100  # Within 2-fold
        metrics['accuracy_5fold'] = np.mean(ratios <= 5) * 100  # Within 5-fold
        
        # Calculate median relative error
        relative_errors = np.abs(y_pred - y_test_true) / y_test_true
        metrics['median_relative_error'] = np.median(relative_errors)
        metrics['mean_relative_error'] = np.mean(relative_errors)
        
        return metrics
    
    def get_feature_importance(self):
        """
        Get feature importance scores.
        
        Returns:
            pd.DataFrame: Feature importance ranked by contribution
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        importances = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
    
    def save(self, filepath):
        """
        Save trained model to disk.
        
        Args:
            filepath (str): Path to save model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'params': self.params,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load(cls, filepath):
        """
        Load trained model from disk.
        
        Args:
            filepath (str): Path to saved model
            
        Returns:
            ABLEModel: Loaded model instance
        """
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(params=model_data['params'])
        
        # Restore state
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.is_trained = model_data['is_trained']
        
        return instance


def load_able_model(filepath):
    """
    Convenience function to load ABLE model.
    
    Args:
        filepath (str): Path to saved model
        
    Returns:
        ABLEModel: Loaded model instance
    """
    return ABLEModel.load(filepath)


def create_geometry_score(r_link, r_area, r_bond):
    """
    Calculate composite geometry score from individual ratios.
    
    Args:
        r_link (float): Linker adequacy ratio
        r_area (float): Contact area ratio
        r_bond (float): Bond number ratio
        
    Returns:
        float: Geometry score (0-3)
    """
    score = 0
    
    # Linker adequacy (0.75-2.0 is optimal)
    if 0.75 <= r_link <= 2.0:
        score += 1
    
    # Contact area balance (0.6-1.3 is optimal)
    if 0.6 <= r_area <= 1.3:
        score += 1
    
    # Bond number balance (0.5-1.2 is optimal)
    if 0.5 <= r_bond <= 1.2:
        score += 1
    
    return score
