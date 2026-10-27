"""
ABLE Model Implementation
AlphaFold-assisted Biparatopic Nanobody Avidity Prediction

v2.0 (2026) — updated to the final published model configuration
(gradient boosting, seed 2 / rank 1 of the 200,000-combination × 7-seed
hyperparameter search on 102 constructs).
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


def calculate_geometry_score(r_link, r_area, r_bond):
    """
    Composite geometry score (continuous, 0–3).

    Each term penalises how far a ratio deviates from its optimum of 1,
    with a hard cap on the penalty:

        S = (1 - min(|R_link - 1|, 1))     # linker adequacy
          + (1 - min(|R_area - 1|, 0.5))   # contact-area balance
          + (1 - min(|R_bond - 1|, 0.5))   # bond-number balance

    This is the exact formula used to build the training data
    (Parameters_3200s.xlsx, column ``Sgeometry``).

    Args:
        r_link (float): linker adequacy ratio, L_link / D_mono
        r_area (float): contact area ratio, A_N / A_C
        r_bond (float): bond number ratio, n_t^C / n_t^N

    Returns:
        float: geometry score in [0, 3]
    """
    return ((1 - min(abs(r_link - 1), 1))
            + (1 - min(abs(r_area - 1), 0.5))
            + (1 - min(abs(r_bond - 1), 0.5)))


class ABLEModel:
    """
    ABLE: AlphaFold-assisted Biparatopic Nanobody Avidity Prediction Model

    Predicts the apparent dissociation constant K_D^bipara of biparatopic
    nanobody constructs from 13 structural and energetic features derived
    from AlphaFold-predicted complexes.

    Feature notes (v2.0):
      * K_D^N and K_D^C enter the model in **natural-log scale** (raw
        values are in nM). ``prepare_data`` converts nM -> ln automatically.
      * The target K_D^bipara is modelled as ln(K_D^bipara); ``predict()``
        back-transforms to nM.
      * S_geometry is the continuous 0-3 score defined above.
    """

    # Model input features, in the exact order the model expects them.
    INPUT_FEATURES = [
        'KD_N',      # ln(K_D of N-terminal monomer)
        'KD_C',      # ln(K_D of C-terminal monomer)
        'D_mono',    # distance between nanobody termini (A)
        'd_epi',     # epitope distance (A, negative on epitope clash)
        'L_link',    # linker contour length (A)
        'R_link',    # L_link / D_mono
        'R_area',    # A_N / A_C
        'R_bond',    # n_t^C / n_t^N
        'AN',        # N-terminal buried SASA (A^2)
        'AC',        # C-terminal buried SASA (A^2)
        'n_bonds_N', # N-terminal interface contacts
        'n_bonds_C', # C-terminal interface contacts
        'S_geometry' # continuous geometry score, 0-3
    ]

    FEATURE_NAMES = INPUT_FEATURES

    # Target column
    TARGET_COL = 'KD_bipara'

    # Optimal hyperparameters (seed 2 / rank 1 of the full grid search;
    # see reproduce/ for the exact search protocol)
    OPTIMAL_PARAMS = {
        'n_estimators': 150,        # budget; early stopping -> 54 trees
        'learning_rate': 0.3,
        'loss': 'huber',
        'alpha': 0.9,
        'subsample': 0.8,
        'max_features': 0.6,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 1,
        'min_impurity_decrease': 0.001,
        'ccp_alpha': 0.005,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10,
        'tol': 0.0001,
        'random_state': 2,
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
        Prepare features and target from a DataFrame.

        K_D^N and K_D^C are converted from nM to natural-log scale when
        supplied as raw nM values (detected heuristically: any value
        > 20 cannot be a natural log of an nM KD). The target K_D^bipara
        is treated the same way.

        Args:
            df (pd.DataFrame): Input data with the 13 feature columns
                (raw nM or log scale both accepted) and the target column
            target_col (str, optional): Target column name

        Returns:
            tuple: (X_features, y_target) as numpy arrays
        """
        if target_col is None:
            target_col = self.TARGET_COL

        missing = set(self.feature_names + [target_col]) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy()

        def to_log(s):
            s = pd.to_numeric(s, errors='coerce')
            if (s > 20).any():
                print("⚠ Converting KD values from nM to natural log scale")
                return np.log(s)
            return s

        df['KD_N'] = to_log(df['KD_N'])
        df['KD_C'] = to_log(df['KD_C'])

        X = df[self.feature_names].apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(df[target_col], errors='coerce')

        # Drop rows with missing values in any feature or the target
        # (the curated parameter table contains unmeasured constructs)
        valid = X.notna().all(axis=1) & y.notna()
        n_dropped = int((~valid).sum())
        if n_dropped:
            print(f"⚠ Dropped {n_dropped} row(s) with missing values (unmeasured constructs)")
        X = X[valid].values.astype(float)
        y = y[valid].values.astype(float)
        if (y > 20).any():
            y = np.log(y)

        print(f"✓ Prepared data: X={X.shape}, y={y.shape}")
        return X, y

    def train(self, X_train, y_train):
        """
        Train the ABLE model.

        Args:
            X_train (np.ndarray): Feature matrix, ln(K_D) scale for KD columns
            y_train (np.ndarray): Target ln(K_D^bipara)
        """
        print("Training ABLE model...")

        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        self.actual_trees_trained = self.model.n_estimators_
        print(f"✓ Model trained with {self.actual_trees_trained} trees")

    def train_from_excel(self, excel_path, test_size=0.2, random_state=456,
                         column_map=None):
        """
        Train model directly from Excel file.

        Args:
            excel_path (str): Path to Excel file
            test_size (float): Proportion for test set
            random_state (int): Random seed for the split
            column_map (dict, optional): Mapping from the file's column
                names to the 13 feature names + 'KD_bipara' expected by
                prepare_data (see examples/01_train_model.py)

        Returns:
            tuple: (X_test, y_test, y_pred) for evaluation
        """
        df = self.load_excel_data(excel_path)
        if column_map is not None:
            df = df.rename(columns=column_map)
        X, y = self.prepare_data(df)

        discretizer = KBinsDiscretizer(
            n_bins=min(10, len(y) // 10),
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

        self.train(X_train, y_train)
        y_pred = self.predict(X_test)

        return X_test, y_test, y_pred

    def predict(self, X):
        """
        Predict biparatopic dissociation constants.

        Args:
            X (np.ndarray or pd.DataFrame): Feature matrix with the 13
                model features in order (KD_N / KD_C in **ln scale** —
                use ``utils.prepare_model_input`` to build this from raw
                measurements)

        Returns:
            np.ndarray: Predicted K_D^bipara values in nM
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values

        X_scaled = self.scaler.transform(X)
        y_pred_log = self.model.predict(X_scaled)
        return np.exp(y_pred_log)

    def evaluate(self, X_test, y_test_true):
        """
        Evaluate model performance.

        Args:
            X_test (np.ndarray): Test features (ln scale KDs)
            y_test_true (np.ndarray): True values, ln(KD) or nM

        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.predict(X_test)

        y_true = np.asarray(y_test_true, dtype=float)
        if (y_true > 20).any():  # nM scale
            y_true_nM = y_true
            y_true_log = np.log(y_true)
        else:
            y_true_log = y_true
            y_true_nM = np.exp(y_true)

        y_pred_log = np.log(y_pred)

        metrics = {
            'r2': r2_score(y_true_log, y_pred_log),
            'rmse': np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
            'mae': mean_absolute_error(y_true_log, y_pred_log),
            'rmse_nM': np.sqrt(mean_squared_error(y_true_nM, y_pred)),
            'mae_nM': mean_absolute_error(y_true_nM, y_pred),
        }

        ratios = np.maximum(y_pred / y_true_nM, y_true_nM / y_pred)
        metrics['accuracy_2fold'] = np.mean(ratios <= 2) * 100
        metrics['accuracy_5fold'] = np.mean(ratios <= 5) * 100

        relative_errors = np.abs(y_pred - y_true_nM) / y_true_nM
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

        instance = cls(params=model_data['params'])

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
