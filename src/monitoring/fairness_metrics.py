"""
Fairness and bias detection metrics for ethical AI monitoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union

from sklearn.metrics import confusion_matrix

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FairnessMonitor:
    """
    Monitor fairness metrics across demographic groups.
    """
    
    def __init__(
        self,
        sensitive_features: List[str],
        threshold: float = 0.1
    ):
        """
        Initialize fairness monitor.
        
        Args:
            sensitive_features: List of sensitive feature names (e.g., 'race', 'gender')
            threshold: Fairness violation threshold
        """
        self.sensitive_features = sensitive_features
        self.threshold = threshold
    
    def demographic_parity_difference(
        self,
        y_pred: np.ndarray,
        sensitive_feature: np.ndarray
    ) -> float:
        """
        Calculate demographic parity difference.
        Measures difference in positive prediction rates between groups.
        
        Args:
            y_pred: Predictions
            sensitive_feature: Sensitive attribute values
            
        Returns:
            Demographic parity difference
        """
        df = pd.DataFrame({
            'pred': y_pred,
            'sensitive': sensitive_feature
        })
        
        # Positive rate by group
        positive_rates = df.groupby('sensitive')['pred'].mean()
        
        # Max difference
        dpd = positive_rates.max() - positive_rates.min()
        
        return float(dpd)
    
    def equal_opportunity_difference(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_feature: np.ndarray
    ) -> float:
        """
        Calculate equal opportunity difference.
        Measures difference in true positive rates between groups.
        
        Args:
            y_true: True labels
            y_pred: Predictions
            sensitive_feature: Sensitive attribute values
            
        Returns:
            Equal opportunity difference
        """
        df = pd.DataFrame({
            'true': y_true,
            'pred': y_pred,
            'sensitive': sensitive_feature
        })
        
        # True positive rate by group
        tpr_by_group = {}
        for group in df['sensitive'].unique():
            group_data = df[df['sensitive'] == group]
            y_true_group = group_data['true']
            y_pred_group = group_data['pred']
            
            # TPR = TP / (TP + FN)
            tp = ((y_true_group == 1) & (y_pred_group == 1)).sum()
            fn = ((y_true_group == 1) & (y_pred_group == 0)).sum()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            tpr_by_group[group] = tpr
        
        # Max difference
        tpr_values = list(tpr_by_group.values())
        eod = max(tpr_values) - min(tpr_values) if tpr_values else 0.0
        
        return float(eod)
    
    def equalized_odds_difference(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_feature: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate equalized odds difference.
        Considers both TPR and FPR differences.
        
        Args:
            y_true: True labels
            y_pred: Predictions
            sensitive_feature: Sensitive attribute values
            
        Returns:
            Dictionary with TPR and FPR differences
        """
        df = pd.DataFrame({
            'true': y_true,
            'pred': y_pred,
            'sensitive': sensitive_feature
        })
        
        tpr_by_group = {}
        fpr_by_group = {}
        
        for group in df['sensitive'].unique():
            group_data = df[df['sensitive'] == group]
            y_true_group = group_data['true']
            y_pred_group = group_data['pred']
            
            # TPR = TP / (TP + FN)
            tp = ((y_true_group == 1) & (y_pred_group == 1)).sum()
            fn = ((y_true_group == 1) & (y_pred_group == 0)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # FPR = FP / (FP + TN)
            fp = ((y_true_group == 0) & (y_pred_group == 1)).sum()
            tn = ((y_true_group == 0) & (y_pred_group == 0)).sum()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            tpr_by_group[group] = tpr
            fpr_by_group[group] = fpr
        
        tpr_values = list(tpr_by_group.values())
        fpr_values = list(fpr_by_group.values())
        
        tpr_diff = max(tpr_values) - min(tpr_values) if tpr_values else 0.0
        fpr_diff = max(fpr_values) - min(fpr_values) if fpr_values else 0.0
        
        return {
            'tpr_difference': float(tpr_diff),
            'fpr_difference': float(fpr_diff),
            'max_difference': float(max(tpr_diff, fpr_diff))
        }
    
    def disparate_impact_ratio(
        self,
        y_pred: np.ndarray,
        sensitive_feature: np.ndarray
    ) -> float:
        """
        Calculate disparate impact ratio.
        Ratio of positive rates between unprivileged and privileged groups.
        
        Args:
            y_pred: Predictions
            sensitive_feature: Sensitive attribute values
            
        Returns:
            Disparate impact ratio
        """
        df = pd.DataFrame({
            'pred': y_pred,
            'sensitive': sensitive_feature
        })
        
        # Positive rate by group
        positive_rates = df.groupby('sensitive')['pred'].mean()
        
        # Ratio of min to max
        if positive_rates.max() > 0:
            dir_ratio = positive_rates.min() / positive_rates.max()
        else:
            dir_ratio = 1.0
        
        return float(dir_ratio)
    
    def group_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_feature: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance metrics for each group.
        
        Args:
            y_true: True labels
            y_pred: Predictions
            sensitive_feature: Sensitive attribute values
            
        Returns:
            Dictionary of metrics by group
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        df = pd.DataFrame({
            'true': y_true,
            'pred': y_pred,
            'sensitive': sensitive_feature
        })
        
        group_metrics = {}
        
        for group in df['sensitive'].unique():
            group_data = df[df['sensitive'] == group]
            y_true_group = group_data['true']
            y_pred_group = group_data['pred']
            
            metrics = {
                'count': len(group_data),
                'accuracy': accuracy_score(y_true_group, y_pred_group),
                'precision': precision_score(y_true_group, y_pred_group, zero_division=0),
                'recall': recall_score(y_true_group, y_pred_group, zero_division=0),
                'f1': f1_score(y_true_group, y_pred_group, zero_division=0),
                'positive_rate': float(y_pred_group.mean())
            }
            
            group_metrics[str(group)] = metrics
        
        return group_metrics
    
    def comprehensive_fairness_report(
        self,
        X: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Generate comprehensive fairness report.
        
        Args:
            X: Features (must include sensitive features)
            y_true: True labels
            y_pred: Predictions
            
        Returns:
            Complete fairness analysis
        """
        logger.info("Generating comprehensive fairness report...")
        
        report = {
            'fairness_metrics': {},
            'group_performance': {},
            'violations': []
        }
        
        for sensitive_feature in self.sensitive_features:
            if sensitive_feature not in X.columns:
                logger.warning(f"Sensitive feature '{sensitive_feature}' not found in data")
                continue
            
            sensitive_values = X[sensitive_feature].values
            
            # Calculate fairness metrics
            dpd = self.demographic_parity_difference(y_pred, sensitive_values)
            eod = self.equal_opportunity_difference(y_true, y_pred, sensitive_values)
            equalized_odds = self.equalized_odds_difference(y_true, y_pred, sensitive_values)
            dir_ratio = self.disparate_impact_ratio(y_pred, sensitive_values)
            
            metrics = {
                'demographic_parity_difference': dpd,
                'equal_opportunity_difference': eod,
                'equalized_odds': equalized_odds,
                'disparate_impact_ratio': dir_ratio
            }
            
            # Check for violations
            if dpd > self.threshold:
                report['violations'].append({
                    'feature': sensitive_feature,
                    'metric': 'demographic_parity_difference',
                    'value': dpd,
                    'threshold': self.threshold
                })
            
            if eod > self.threshold:
                report['violations'].append({
                    'feature': sensitive_feature,
                    'metric': 'equal_opportunity_difference',
                    'value': eod,
                    'threshold': self.threshold
                })
            
            if dir_ratio < (1 - self.threshold):
                report['violations'].append({
                    'feature': sensitive_feature,
                    'metric': 'disparate_impact_ratio',
                    'value': dir_ratio,
                    'threshold': 1 - self.threshold
                })
            
            report['fairness_metrics'][sensitive_feature] = metrics
            
            # Group performance
            group_perf = self.group_metrics(y_true, y_pred, sensitive_values)
            report['group_performance'][sensitive_feature] = group_perf
        
        report['fairness_violations_detected'] = len(report['violations']) > 0
        report['total_violations'] = len(report['violations'])
        
        if report['fairness_violations_detected']:
            logger.warning(f"Fairness violations detected: {report['total_violations']}")
        else:
            logger.info("No fairness violations detected")
        
        return report


def calculate_fairness_metrics(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: List[str],
    threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Quick function to calculate fairness metrics.
    
    Args:
        X: Features
        y_true: True labels
        y_pred: Predictions
        sensitive_features: List of sensitive features
        threshold: Fairness threshold
        
    Returns:
        Fairness metrics
    """
    monitor = FairnessMonitor(sensitive_features, threshold)
    return monitor.comprehensive_fairness_report(X, y_true, y_pred)


__all__ = [
    "FairnessMonitor",
    "calculate_fairness_metrics"
]

