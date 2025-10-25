"""
Data drift detection metrics and statistical tests.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
from scipy.spatial.distance import wasserstein_distance

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DriftDetector:
    """
    Comprehensive drift detection for features and predictions.
    """
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        threshold: float = 0.05
    ):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference/baseline data
            threshold: P-value threshold for statistical tests
        """
        self.reference_data = reference_data
        self.threshold = threshold
        
        # Store statistics of reference data
        self.reference_stats = self._compute_statistics(reference_data)
    
    def _compute_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for data."""
        stats_dict = {}
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                stats_dict[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'median': data[col].median(),
                    'q25': data[col].quantile(0.25),
                    'q75': data[col].quantile(0.75)
                }
            else:
                stats_dict[col] = {
                    'value_counts': data[col].value_counts().to_dict(),
                    'unique_count': data[col].nunique()
                }
        
        return stats_dict
    
    def kolmogorov_smirnov_test(
        self,
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform Kolmogorov-Smirnov test for numerical features.
        
        Args:
            current_data: Current/production data
            features: List of features to test (None = all numerical)
            
        Returns:
            Dictionary of {feature: {'statistic': ..., 'p_value': ...}}
        """
        logger.info("Running Kolmogorov-Smirnov test...")
        
        if features is None:
            features = self.reference_data.select_dtypes(include=[np.number]).columns.tolist()
        
        results = {}
        
        for feature in features:
            if feature not in self.reference_data.columns or feature not in current_data.columns:
                continue
            
            if pd.api.types.is_numeric_dtype(self.reference_data[feature]):
                ref_values = self.reference_data[feature].dropna()
                curr_values = current_data[feature].dropna()
                
                statistic, p_value = stats.ks_2samp(ref_values, curr_values)
                
                results[feature] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'drift_detected': p_value < self.threshold
                }
        
        drift_count = sum(1 for r in results.values() if r['drift_detected'])
        logger.info(f"KS Test: {drift_count}/{len(results)} features show drift")
        
        return results
    
    def chi_square_test(
        self,
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform Chi-Square test for categorical features.
        
        Args:
            current_data: Current/production data
            features: List of features to test (None = all categorical)
            
        Returns:
            Dictionary of {feature: {'statistic': ..., 'p_value': ...}}
        """
        logger.info("Running Chi-Square test...")
        
        if features is None:
            features = self.reference_data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        results = {}
        
        for feature in features:
            if feature not in self.reference_data.columns or feature not in current_data.columns:
                continue
            
            # Create contingency table
            ref_counts = self.reference_data[feature].value_counts()
            curr_counts = current_data[feature].value_counts()
            
            # Align categories
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            ref_counts = ref_counts.reindex(all_categories, fill_value=0)
            curr_counts = curr_counts.reindex(all_categories, fill_value=0)
            
            # Chi-square test
            contingency = np.array([ref_counts.values, curr_counts.values])
            
            try:
                statistic, p_value, _, _ = stats.chi2_contingency(contingency)
                
                results[feature] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'drift_detected': p_value < self.threshold
                }
            except Exception as e:
                logger.warning(f"Chi-square test failed for {feature}: {e}")
                continue
        
        drift_count = sum(1 for r in results.values() if r['drift_detected'])
        logger.info(f"Chi-Square Test: {drift_count}/{len(results)} features show drift")
        
        return results
    
    def population_stability_index(
        self,
        current_data: pd.DataFrame,
        feature: str,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        Args:
            current_data: Current/production data
            feature: Feature name
            n_bins: Number of bins for discretization
            
        Returns:
            PSI value
        """
        if feature not in self.reference_data.columns or feature not in current_data.columns:
            raise ValueError(f"Feature {feature} not found in data")
        
        ref_values = self.reference_data[feature].dropna()
        curr_values = current_data[feature].dropna()
        
        # Discretize if numerical
        if pd.api.types.is_numeric_dtype(ref_values):
            bins = np.linspace(
                min(ref_values.min(), curr_values.min()),
                max(ref_values.max(), curr_values.max()),
                n_bins + 1
            )
            
            ref_binned = np.digitize(ref_values, bins)
            curr_binned = np.digitize(curr_values, bins)
        else:
            # For categorical, use value counts
            ref_binned = ref_values
            curr_binned = curr_values
        
        # Calculate distributions
        ref_dist = pd.Series(ref_binned).value_counts(normalize=True).sort_index()
        curr_dist = pd.Series(curr_binned).value_counts(normalize=True).sort_index()
        
        # Align indices
        all_bins = set(ref_dist.index) | set(curr_dist.index)
        ref_dist = ref_dist.reindex(all_bins, fill_value=0.0001)  # Small value to avoid log(0)
        curr_dist = curr_dist.reindex(all_bins, fill_value=0.0001)
        
        # Calculate PSI
        psi = np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist))
        
        return float(psi)
    
    def wasserstein_distance_metric(
        self,
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate Wasserstein distance for numerical features.
        
        Args:
            current_data: Current/production data
            features: List of features (None = all numerical)
            
        Returns:
            Dictionary of {feature: distance}
        """
        logger.info("Calculating Wasserstein distances...")
        
        if features is None:
            features = self.reference_data.select_dtypes(include=[np.number]).columns.tolist()
        
        results = {}
        
        for feature in features:
            if feature not in self.reference_data.columns or feature not in current_data.columns:
                continue
            
            if pd.api.types.is_numeric_dtype(self.reference_data[feature]):
                ref_values = self.reference_data[feature].dropna().values
                curr_values = current_data[feature].dropna().values
                
                distance = wasserstein_distance(ref_values, curr_values)
                results[feature] = float(distance)
        
        return results
    
    def detect_drift(
        self,
        current_data: pd.DataFrame,
        methods: List[str] = ['ks', 'chi2', 'psi']
    ) -> Dict[str, Any]:
        """
        Comprehensive drift detection using multiple methods.
        
        Args:
            current_data: Current/production data
            methods: List of methods to use ('ks', 'chi2', 'psi', 'wasserstein')
            
        Returns:
            Dictionary with all drift detection results
        """
        logger.info("Running comprehensive drift detection...")
        
        results = {}
        
        if 'ks' in methods:
            results['kolmogorov_smirnov'] = self.kolmogorov_smirnov_test(current_data)
        
        if 'chi2' in methods:
            results['chi_square'] = self.chi_square_test(current_data)
        
        if 'psi' in methods:
            psi_results = {}
            for col in current_data.columns:
                if col in self.reference_data.columns:
                    try:
                        psi_results[col] = self.population_stability_index(current_data, col)
                    except Exception as e:
                        logger.warning(f"PSI calculation failed for {col}: {e}")
            results['psi'] = psi_results
        
        if 'wasserstein' in methods:
            results['wasserstein'] = self.wasserstein_distance_metric(current_data)
        
        # Summary
        drift_detected = False
        if 'kolmogorov_smirnov' in results:
            drift_detected |= any(r['drift_detected'] for r in results['kolmogorov_smirnov'].values())
        if 'chi_square' in results:
            drift_detected |= any(r['drift_detected'] for r in results['chi_square'].values())
        
        results['summary'] = {
            'drift_detected': drift_detected,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"Drift detection complete. Drift detected: {drift_detected}")
        
        return results


def detect_prediction_drift(
    reference_predictions: np.ndarray,
    current_predictions: np.ndarray,
    threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Detect drift in model predictions.
    
    Args:
        reference_predictions: Baseline predictions
        current_predictions: Current predictions
        threshold: Statistical threshold
        
    Returns:
        Drift detection results
    """
    logger.info("Detecting prediction drift...")
    
    # KS test for numerical predictions
    if len(np.unique(reference_predictions)) > 2:
        statistic, p_value = stats.ks_2samp(reference_predictions, current_predictions)
    else:
        # Chi-square for binary predictions
        ref_counts = pd.Series(reference_predictions).value_counts()
        curr_counts = pd.Series(current_predictions).value_counts()
        
        all_values = set(ref_counts.index) | set(curr_counts.index)
        ref_counts = ref_counts.reindex(all_values, fill_value=0)
        curr_counts = curr_counts.reindex(all_values, fill_value=0)
        
        contingency = np.array([ref_counts.values, curr_counts.values])
        statistic, p_value, _, _ = stats.chi2_contingency(contingency)
    
    results = {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'drift_detected': p_value < threshold,
        'reference_mean': float(np.mean(reference_predictions)),
        'current_mean': float(np.mean(current_predictions)),
        'reference_std': float(np.std(reference_predictions)),
        'current_std': float(np.std(current_predictions))
    }
    
    logger.info(f"Prediction drift detected: {results['drift_detected']}")
    
    return results


__all__ = [
    "DriftDetector",
    "detect_prediction_drift"
]

