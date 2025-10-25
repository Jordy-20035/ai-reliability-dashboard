"""
Performance monitoring metrics for model reliability.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import time
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, mean_squared_error
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceMonitor:
    """
    Monitor model performance metrics over time.
    """
    
    def __init__(
        self,
        model: Any,
        baseline_metrics: Optional[Dict[str, float]] = None,
        threshold: float = 0.05
    ):
        """
        Initialize performance monitor.
        
        Args:
            model: Trained model
            baseline_metrics: Baseline performance metrics
            threshold: Degradation threshold (e.g., 0.05 = 5% drop)
        """
        self.model = model
        self.baseline_metrics = baseline_metrics or {}
        self.threshold = threshold
        self.history: List[Dict[str, Any]] = []
    
    def calculate_metrics(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        include_probabilities: bool = True
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            X: Features
            y: True labels
            include_probabilities: Whether to calculate probability-based metrics
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Calculating performance metrics...")
        
        # Predictions
        y_pred = self.model.predict(X)
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y, y_pred, average='binary', zero_division=0),
            'n_samples': len(y),
            'positive_rate': float(y_pred.mean()),
            'timestamp': datetime.now().isoformat()
        }
        
        # Probability-based metrics
        if include_probabilities and hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X)[:, 1]
            
            metrics['roc_auc'] = roc_auc_score(y, y_proba)
            metrics['log_loss'] = log_loss(y, y_proba)
            
            # Calibration metrics
            metrics['mean_predicted_probability'] = float(y_proba.mean())
            metrics['prediction_std'] = float(y_proba.std())
        
        return metrics
    
    def measure_latency(
        self,
        X: pd.DataFrame,
        n_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Measure prediction latency.
        
        Args:
            X: Features
            n_iterations: Number of iterations for averaging
            
        Returns:
            Latency statistics
        """
        logger.info("Measuring prediction latency...")
        
        latencies = []
        
        for _ in range(n_iterations):
            start_time = time.time()
            _ = self.model.predict(X)
            end_time = time.time()
            
            latencies.append(end_time - start_time)
        
        latencies = np.array(latencies)
        
        metrics = {
            'mean_latency_seconds': float(latencies.mean()),
            'median_latency_seconds': float(np.median(latencies)),
            'std_latency_seconds': float(latencies.std()),
            'min_latency_seconds': float(latencies.min()),
            'max_latency_seconds': float(latencies.max()),
            'p95_latency_seconds': float(np.percentile(latencies, 95)),
            'p99_latency_seconds': float(np.percentile(latencies, 99)),
            'throughput_per_second': float(1.0 / latencies.mean())
        }
        
        logger.info(f"Mean latency: {metrics['mean_latency_seconds']*1000:.2f}ms")
        
        return metrics
    
    def detect_degradation(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Detect performance degradation.
        
        Args:
            current_metrics: Current performance metrics
            baseline_metrics: Baseline to compare against
            
        Returns:
            Degradation analysis
        """
        if baseline_metrics is None:
            baseline_metrics = self.baseline_metrics
        
        if not baseline_metrics:
            logger.warning("No baseline metrics available for comparison")
            return {'degradation_detected': False, 'reason': 'No baseline'}
        
        degradation_results = {}
        degradation_detected = False
        
        key_metrics = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
        
        for metric in key_metrics:
            if metric in current_metrics and metric in baseline_metrics:
                baseline_val = baseline_metrics[metric]
                current_val = current_metrics[metric]
                
                # Calculate relative change
                if baseline_val > 0:
                    relative_change = (current_val - baseline_val) / baseline_val
                else:
                    relative_change = 0.0
                
                # Check for degradation (negative change exceeding threshold)
                is_degraded = relative_change < -self.threshold
                
                degradation_results[metric] = {
                    'baseline': baseline_val,
                    'current': current_val,
                    'absolute_change': current_val - baseline_val,
                    'relative_change': relative_change,
                    'degraded': is_degraded
                }
                
                if is_degraded:
                    degradation_detected = True
        
        results = {
            'degradation_detected': degradation_detected,
            'threshold': self.threshold,
            'metrics': degradation_results,
            'timestamp': datetime.now().isoformat()
        }
        
        if degradation_detected:
            degraded_metrics = [m for m, v in degradation_results.items() if v['degraded']]
            logger.warning(f"Performance degradation detected in: {degraded_metrics}")
        else:
            logger.info("No performance degradation detected")
        
        return results
    
    def monitor(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        store_history: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive monitoring including metrics, latency, and degradation.
        
        Args:
            X: Features
            y: True labels
            store_history: Whether to store results in history
            
        Returns:
            Complete monitoring results
        """
        logger.info("Running comprehensive performance monitoring...")
        
        # Calculate metrics
        metrics = self.calculate_metrics(X, y)
        
        # Measure latency
        latency = self.measure_latency(X[:min(100, len(X))])
        
        # Detect degradation
        degradation = self.detect_degradation(metrics)
        
        results = {
            'metrics': metrics,
            'latency': latency,
            'degradation': degradation,
            'timestamp': datetime.now().isoformat()
        }
        
        if store_history:
            self.history.append(results)
        
        return results
    
    def get_history_dataframe(self) -> pd.DataFrame:
        """
        Get monitoring history as DataFrame.
        
        Returns:
            DataFrame with historical metrics
        """
        if not self.history:
            return pd.DataFrame()
        
        records = []
        for entry in self.history:
            record = entry['metrics'].copy()
            record.update({
                f"latency_{k}": v 
                for k, v in entry['latency'].items()
            })
            record['degradation_detected'] = entry['degradation']['degradation_detected']
            records.append(record)
        
        return pd.DataFrame(records)


def calculate_performance_metrics(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series
) -> Dict[str, float]:
    """
    Quick function to calculate performance metrics.
    
    Args:
        model: Trained model
        X: Features
        y: True labels
        
    Returns:
        Dictionary of metrics
    """
    monitor = PerformanceMonitor(model)
    return monitor.calculate_metrics(X, y)


def calculate_business_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    cost_false_positive: float = 1.0,
    cost_false_negative: float = 5.0,
    revenue_true_positive: float = 10.0
) -> Dict[str, float]:
    """
    Calculate business-oriented metrics.
    
    Args:
        y_true: True labels
        y_pred: Predictions
        y_proba: Prediction probabilities
        cost_false_positive: Cost of false positive
        cost_false_negative: Cost of false negative
        revenue_true_positive: Revenue from true positive
        
    Returns:
        Business metrics
    """
    from sklearn.metrics import confusion_matrix
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    total_cost = (fp * cost_false_positive) + (fn * cost_false_negative)
    total_revenue = tp * revenue_true_positive
    net_value = total_revenue - total_cost
    
    metrics = {
        'total_cost': float(total_cost),
        'total_revenue': float(total_revenue),
        'net_value': float(net_value),
        'cost_per_prediction': float(total_cost / len(y_true)),
        'revenue_per_prediction': float(total_revenue / len(y_true)),
        'roi': float(net_value / total_cost) if total_cost > 0 else 0.0
    }
    
    return metrics


__all__ = [
    "PerformanceMonitor",
    "calculate_performance_metrics",
    "calculate_business_metrics"
]

