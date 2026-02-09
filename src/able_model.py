"""
ABLE Model Implementation
AlphaFold-assisted Biparatopic Nanobody Avidity Prediction
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
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
    
    # Required features for prediction (in exact order as in dataset)
    FEATURE_NAMES = [
        'KD_N', 'KD_C', 'KD_bipara', 'D_mono', 'd_epi', 'L_link',
        'R_link', 'R_area', 'R_bond', 'AN', 'AC', 'n_bonds_N',
        'n_bonds_C', 'S_geometry'
    ]
    
    # Input features (exclude target)
    INPUT_FEATURES = FEATURE_NAMES[:-1]  # All except KD_bipara
    
    # Target column
    TARGET_COL = 'KD_bipara'
    
    # Optimal hyperparameters determined by randomized search
    OPTIMAL_PARAMS = {
        'n_estimators': 100,
        'learning_rate': 0.2,
        'loss': 'absolute_error',
        'criterion': 'friedman_mse',
        'min_samples_split': 4,
        'min_samples_leaf': 2,
        'subsample': 0.95,
        'max_depth': 12,
        'max_features': 0.7,
        'min_impurity_decrease': 0.01,
        'ccp_alpha': 0.01,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10,
        'tol': 0.0001,
        'random_state': 3,
        'verbose': 0
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
        self.feature_names = self.INPUT_FEATURES
    
    def load_excel_data(self, excel_path, sheet_name=0):
        """
        Load data from Excel file.
        
        Args:
            excel_path (str): Path to Excel file
            sheet_name (str/int): Sheet name or index
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            print(f"✓ Loaded data from {excel_path}")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"✗ Error loading Excel file: {e}")
            raise
    
    def prepare_data(self, df, target_col=None):
        """
        Prepare features and target from DataFrame.
        
        Args:
            df (pd.DataFrame): Input data
            target_col (str, optional): Target column name
            
        Returns:
            tuple: (X_features, y_target) as numpy arrays
        """
        if target_col is None:
            target_col = self.TARGET_COL
        
        # Check required features
        missing_features = set(self.feature_names + [target_col]) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required columns: {missing_features}")
        
        # Extract features
        X = df[self.feature_names].values
        
        # Extract target (assuming already in natural log scale)
        y = df[target_col].values
        
        # If target is in nM scale, convert to log
        if np.max(y) > 100:  # Assuming values > 100 are in nM scale
            print("⚠ Converting target from nM to natural log scale")
            y = np.log(y)
        
        print(f"✓ Prepared data: X={X.shape}, y={y.shape}")
        return X, y
    
    def train(self, X_train, y_train):
        """
        Train the ABLE model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target (ln(KD))
        """
        print("Training ABLE model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Store actual number of trees trained
        self.actual_trees_trained = self.model.n_estimators_
        print(f"✓ Model trained with {self.actual_trees_trained} trees")
    
    def train_from_excel(self, excel_path, test_size=0.2, random_state=456):
        """
        Train model directly from Excel file.
        
        Args:
            excel_path (str): Path to Excel file
            test_size (float): Proportion for test set
            random_state (int): Random seed
            
        Returns:
            tuple: (X_test, y_test, y_pred) for evaluation
        """
        # Load data
        df = self.load_excel_data(excel_path)
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Create stratified split (preserve distribution of target)
        discretizer = KBinsDiscretizer(
            n_bins=min(10, len(y)//10), 
            encode='ordinal', 
            strategy='quantile'
        )
        y_binned = discretizer.fit_transform(y.reshape(-1, 1)).flatten().astype(int)
        
        sss = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=test_size, 
            random_state=random_state
        )
        
        for train_idx, test_idx in sss.split(X, y_binned):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        
        print(f"✓ Data split: Train={len(X_train)}, Test={len(X_test)}")
        
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
            y_test_true (np.ndarray): True KD values in nM (or ln(KD))
            
        Returns:
            dict: Evaluation metrics
        """
        # Predict
        y_pred = self.predict(X_test)
        
        # Convert y_test to appropriate scale
        if np.max(y_test_true) > 100:  # Assuming nM scale
            y_test_nM = y_test_true
            y_test_log = np.log(y_test_true)
        else:  # Already in log scale
            y_test_log = y_test_true
            y_test_nM = np.exp(y_test_true)
        
        y_pred_log = np.log(y_pred)
        
        # Calculate metrics
        metrics = {
            'r2': r2_score(y_test_log, y_pred_log),
            'rmse': np.sqrt(mean_squared_error(y_test_log, y_pred_log)),
            'mae': mean_absolute_error(y_test_log, y_pred_log),
            'r2_nM': r2_score(y_test_nM, y_pred),
            'rmse_nM': np.sqrt(mean_squared_error(y_test_nM, y_pred)),
            'mae_nM': mean_absolute_error(y_test_nM, y_pred),
        }
        
        # Calculate fold accuracy
        ratios = np.maximum(y_pred / y_test_nM, y_test_nM / y_pred)
        metrics['accuracy_2fold'] = np.mean(ratios <= 2) * 100
        metrics['accuracy_5fold'] = np.mean(ratios <= 5) * 100
        
        # Calculate relative error
        relative_errors = np.abs(y_pred - y_test_nM) / y_test_nM
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
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Add cumulative importance
        importance_df['cumulative'] = importance_df['importance'].cumsum()
        
        return importance_df
    
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
            'is_trained': self.is_trained,
            'actual_trees_trained': getattr(self, 'actual_trees_trained', 0)
        }
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")
    
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
        instance.actual_trees_trained = model_data.get('actual_trees_trained', 0)
        
        print(f"✓ Model loaded from {filepath}")
        print(f"  Trained: {instance.is_trained}")
        print(f"  Features: {len(instance.feature_names)}")
        
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


def calculate_geometry_score(r_link, r_area, r_bond):
    """
    Calculate composite geometry score from individual ratios.
    
    Args:
        r_link (float): Linker adequacy ratio
        r_area (float): Contact area ratio
        r_bond (float): Bond number ratio
        
    Returns:
        int: Geometry score (0-3)
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
