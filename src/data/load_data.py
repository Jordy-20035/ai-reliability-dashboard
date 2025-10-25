"""
Data loading utilities for various datasets.
Supports Adult Income, COMPAS, and synthetic datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)


def load_adult_data(
    test_size: float = 0.2,
    random_state: int = 42,
    from_cache: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load and split the Adult Income dataset from UCI ML Repository.
    
    Args:
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
        from_cache: Whether to load from cached file if available
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info("Loading Adult Income dataset...")
    
    cache_path = config.data.raw_data_dir / "adult.pkl"
    
    if from_cache and cache_path.exists():
        logger.info(f"Loading from cache: {cache_path}")
        df = pd.read_pickle(cache_path)
    else:
        try:
            # Fetch from OpenML
            adult = fetch_openml('adult', version=2, as_frame=True, parser='auto')
            df = adult.frame
            
            # Save to cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_pickle(cache_path)
            logger.info(f"Saved to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Failed to fetch Adult dataset: {e}")
            raise
    
    # Target column
    target_col = 'class' if 'class' in df.columns else df.columns[-1]
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Convert target to binary (>50K = 1, <=50K = 0)
    y = y.apply(lambda x: 1 if '>50' in str(x) else 0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Dataset loaded: Train={len(X_train)}, Test={len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def load_compas_data(
    filepath: Optional[Path] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load COMPAS recidivism dataset.
    
    Args:
        filepath: Path to COMPAS CSV file
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info("Loading COMPAS dataset...")
    
    if filepath is None:
        filepath = config.data.raw_data_dir / "compas-scores-two-years.csv"
    
    if not filepath.exists():
        logger.warning(f"COMPAS file not found: {filepath}")
        logger.info("Please download from: https://github.com/propublica/compas-analysis")
        raise FileNotFoundError(f"COMPAS dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Select relevant columns
    columns = [
        'age', 'c_charge_degree', 'race', 'age_cat', 'score_text',
        'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score',
        'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out'
    ]
    
    # Use available columns
    available_cols = [col for col in columns if col in df.columns]
    df = df[available_cols]
    
    # Target: two_year_recid or is_recid
    if 'two_year_recid' in df.columns:
        target_col = 'two_year_recid'
    elif 'is_recid' in df.columns:
        target_col = 'is_recid'
    else:
        raise ValueError("Target column not found in COMPAS dataset")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"COMPAS loaded: Train={len(X_train)}, Test={len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def generate_synthetic_data(
    n_samples: int = 10000,
    n_features: int = 10,
    n_informative: int = 5,
    n_redundant: int = 2,
    flip_y: float = 0.1,
    test_size: float = 0.2,
    random_state: int = 42,
    add_demographics: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Generate synthetic classification dataset with optional demographic features.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_informative: Number of informative features
        n_redundant: Number of redundant features
        flip_y: Proportion of label noise
        test_size: Proportion of test set
        random_state: Random seed
        add_demographics: Whether to add demographic features for fairness testing
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info("Generating synthetic dataset...")
    
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        flip_y=flip_y,
        random_state=random_state
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    
    if add_demographics:
        # Add synthetic demographic features
        np.random.seed(random_state)
        df['age'] = np.random.randint(18, 80, n_samples)
        df['gender'] = np.random.choice(['Male', 'Female'], n_samples)
        df['race'] = np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], n_samples)
    
    X = df
    y = pd.Series(y, name='target')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Synthetic data generated: Train={len(X_train)}, Test={len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def load_dataset(
    dataset_name: str = "adult",
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Universal data loader for different datasets.
    
    Args:
        dataset_name: Name of dataset ('adult', 'compas', 'synthetic')
        **kwargs: Additional arguments for specific loaders
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "adult":
        return load_adult_data(**kwargs)
    elif dataset_name == "compas":
        return load_compas_data(**kwargs)
    elif dataset_name == "synthetic":
        return generate_synthetic_data(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    dataset_name: str = "dataset"
) -> None:
    """
    Save processed data to disk.
    
    Args:
        X_train, X_test, y_train, y_test: Data splits
        dataset_name: Name for saved files
    """
    output_dir = config.data.processed_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    X_train.to_pickle(output_dir / f"{dataset_name}_X_train.pkl")
    X_test.to_pickle(output_dir / f"{dataset_name}_X_test.pkl")
    y_train.to_pickle(output_dir / f"{dataset_name}_y_train.pkl")
    y_test.to_pickle(output_dir / f"{dataset_name}_y_test.pkl")
    
    logger.info(f"Processed data saved to {output_dir}")


def load_processed_data(
    dataset_name: str = "dataset"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load previously processed data from disk.
    
    Args:
        dataset_name: Name of saved files
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    input_dir = config.data.processed_data_dir
    
    X_train = pd.read_pickle(input_dir / f"{dataset_name}_X_train.pkl")
    X_test = pd.read_pickle(input_dir / f"{dataset_name}_X_test.pkl")
    y_train = pd.read_pickle(input_dir / f"{dataset_name}_y_train.pkl")
    y_test = pd.read_pickle(input_dir / f"{dataset_name}_y_test.pkl")
    
    logger.info(f"Processed data loaded from {input_dir}")
    
    return X_train, X_test, y_train, y_test


__all__ = [
    "load_adult_data",
    "load_compas_data",
    "generate_synthetic_data",
    "load_dataset",
    "save_processed_data",
    "load_processed_data"
]

