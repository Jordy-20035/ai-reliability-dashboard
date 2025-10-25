"""
Data preprocessing utilities for feature engineering and transformation.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline.
    Handles numerical and categorical features separately.
    """
    
    def __init__(
        self,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        scale_numerical: bool = True,
        encode_categorical: str = "onehot",  # "onehot" or "label"
        handle_missing: str = "mean"  # "mean", "median", "mode", "drop"
    ):
        """
        Initialize preprocessor.
        
        Args:
            numerical_features: List of numerical column names
            categorical_features: List of categorical column names
            scale_numerical: Whether to standardize numerical features
            encode_categorical: Encoding method for categorical features
            handle_missing: Strategy for handling missing values
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.scale_numerical = scale_numerical
        self.encode_categorical = encode_categorical
        self.handle_missing = handle_missing
        
        self.preprocessor: Optional[ColumnTransformer] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names_out: Optional[List[str]] = None
        
    def _detect_feature_types(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Automatically detect numerical and categorical features.
        
        Args:
            X: Input DataFrame
            
        Returns:
            numerical_features, categorical_features
        """
        numerical = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        return numerical, categorical
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            X: Training features
            y: Training target (unused, for sklearn compatibility)
            
        Returns:
            self
        """
        logger.info("Fitting preprocessor...")
        
        # Detect feature types if not provided
        if self.numerical_features is None or self.categorical_features is None:
            num_feats, cat_feats = self._detect_feature_types(X)
            
            if self.numerical_features is None:
                self.numerical_features = num_feats
            if self.categorical_features is None:
                self.categorical_features = cat_feats
        
        logger.info(f"Numerical features: {len(self.numerical_features)}")
        logger.info(f"Categorical features: {len(self.categorical_features)}")
        
        # Build preprocessing pipelines
        transformers = []
        
        # Numerical pipeline
        if self.numerical_features:
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy=self.handle_missing if self.handle_missing != 'mode' else 'mean')),
            ])
            
            if self.scale_numerical:
                num_pipeline.steps.append(('scaler', StandardScaler()))
            
            transformers.append(('num', num_pipeline, self.numerical_features))
        
        # Categorical pipeline
        if self.categorical_features:
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
            ])
            
            if self.encode_categorical == "onehot":
                cat_pipeline.steps.append(
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                )
            
            transformers.append(('cat', cat_pipeline, self.categorical_features))
        
        # Create column transformer
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        
        # Fit the preprocessor
        self.preprocessor.fit(X)
        
        # Store feature names for transformed data
        self._compute_feature_names(X)
        
        logger.info("Preprocessor fitted successfully")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Input features
            
        Returns:
            Transformed DataFrame
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        X_transformed = self.preprocessor.transform(X)
        
        # Convert back to DataFrame with proper column names
        if self.feature_names_out:
            X_transformed = pd.DataFrame(
                X_transformed,
                columns=self.feature_names_out,
                index=X.index
            )
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X: Input features
            y: Training target (unused)
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X)
    
    def _compute_feature_names(self, X: pd.DataFrame) -> None:
        """
        Compute output feature names after transformation.
        
        Args:
            X: Original DataFrame
        """
        feature_names = []
        
        # Numerical features
        if self.numerical_features:
            feature_names.extend(self.numerical_features)
        
        # Categorical features
        if self.categorical_features and self.encode_categorical == "onehot":
            try:
                # Get feature names from OneHotEncoder
                cat_pipeline = None
                for name, pipeline, features in self.preprocessor.transformers_:
                    if name == 'cat':
                        cat_pipeline = pipeline
                        break
                
                if cat_pipeline and hasattr(cat_pipeline.named_steps['encoder'], 'get_feature_names_out'):
                    cat_features = cat_pipeline.named_steps['encoder'].get_feature_names_out(self.categorical_features)
                    feature_names.extend(cat_features)
                else:
                    # Fallback
                    feature_names.extend(self.categorical_features)
            except Exception as e:
                logger.warning(f"Could not get categorical feature names: {e}")
                feature_names.extend(self.categorical_features)
        elif self.categorical_features:
            feature_names.extend(self.categorical_features)
        
        self.feature_names_out = feature_names


def create_drift_data(
    X: pd.DataFrame,
    drift_intensity: float = 0.3,
    features_to_drift: Optional[List[str]] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create drifted version of dataset for testing monitoring.
    
    Args:
        X: Original data
        drift_intensity: Intensity of drift (0-1)
        features_to_drift: Specific features to drift (None = all numerical)
        random_state: Random seed
        
    Returns:
        Drifted DataFrame
    """
    logger.info(f"Creating drifted data with intensity={drift_intensity}")
    
    X_drift = X.copy()
    np.random.seed(random_state)
    
    # Select features to drift
    if features_to_drift is None:
        features_to_drift = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    for feature in features_to_drift:
        if feature in X.columns:
            if X[feature].dtype in ['int64', 'float64']:
                # Add drift to numerical features
                mean = X[feature].mean()
                std = X[feature].std()
                drift = np.random.normal(0, std * drift_intensity, len(X))
                X_drift[feature] = X[feature] + drift
            elif X[feature].dtype == 'object':
                # Randomly change some categorical values
                n_changes = int(len(X) * drift_intensity)
                indices = np.random.choice(len(X), n_changes, replace=False)
                unique_vals = X[feature].unique()
                if len(unique_vals) > 1:
                    X_drift.loc[X_drift.index[indices], feature] = np.random.choice(unique_vals, n_changes)
    
    logger.info(f"Drift applied to {len(features_to_drift)} features")
    
    return X_drift


def balance_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "smote",
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Balance dataset using resampling techniques.
    
    Args:
        X: Features
        y: Target
        method: Balancing method ('smote', 'undersample', 'oversample')
        random_state: Random seed
        
    Returns:
        X_balanced, y_balanced
    """
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    
    logger.info(f"Balancing dataset using {method}")
    logger.info(f"Original class distribution: {y.value_counts().to_dict()}")
    
    if method == "smote":
        sampler = SMOTE(random_state=random_state)
    elif method == "oversample":
        sampler = RandomOverSampler(random_state=random_state)
    elif method == "undersample":
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError(f"Unknown balancing method: {method}")
    
    X_balanced, y_balanced = sampler.fit_resample(X, y)
    
    logger.info(f"Balanced class distribution: {pd.Series(y_balanced).value_counts().to_dict()}")
    
    return X_balanced, y_balanced


__all__ = [
    "DataPreprocessor",
    "create_drift_data",
    "balance_dataset"
]

