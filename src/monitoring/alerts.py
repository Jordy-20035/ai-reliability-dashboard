"""
Alert system for monitoring issues and degradation.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    SUCCESS = "success"


class Alert:
    """Represents a single alert."""
    
    def __init__(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        metric_name: Optional[str] = None,
        current_value: Optional[float] = None,
        baseline_value: Optional[float] = None,
        threshold: Optional[float] = None,
        timestamp: Optional[str] = None
    ):
        """
        Initialize alert.
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity level
            metric_name: Name of metric that triggered alert
            current_value: Current metric value
            baseline_value: Baseline metric value
            threshold: Threshold value
            timestamp: Alert timestamp
        """
        self.title = title
        self.message = message
        self.severity = severity
        self.metric_name = metric_name
        self.current_value = current_value
        self.baseline_value = baseline_value
        self.threshold = threshold
        self.timestamp = timestamp or datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'title': self.title,
            'message': self.message,
            'severity': self.severity.value,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'baseline_value': self.baseline_value,
            'threshold': self.threshold,
            'timestamp': self.timestamp
        }


class AlertManager:
    """Manages alerts for monitoring system."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
    
    def add_alert(self, alert: Alert) -> None:
        """
        Add alert to active alerts.
        
        Args:
            alert: Alert to add
        """
        self.alerts.append(alert)
        self.alert_history.append(alert)
        logger.warning(f"Alert: [{alert.severity.value.upper()}] {alert.title} - {alert.message}")
    
    def clear_alerts(self) -> None:
        """Clear all active alerts."""
        self.alerts.clear()
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return self.alerts.copy()
    
    def get_critical_alerts(self) -> List[Alert]:
        """Get only critical alerts."""
        return [a for a in self.alerts if a.severity == AlertSeverity.CRITICAL]
    
    def get_warning_alerts(self) -> List[Alert]:
        """Get warning and critical alerts."""
        return [a for a in self.alerts if a.severity in [AlertSeverity.WARNING, AlertSeverity.CRITICAL]]
    
    def check_performance_degradation(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        threshold: float = 0.05
    ) -> List[Alert]:
        """
        Check for performance degradation and generate alerts.
        
        Args:
            current_metrics: Current performance metrics
            baseline_metrics: Baseline metrics
            threshold: Degradation threshold (0.05 = 5% drop)
            
        Returns:
            List of alerts
        """
        alerts = []
        key_metrics = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
        
        for metric in key_metrics:
            if metric in current_metrics and metric in baseline_metrics:
                baseline_val = baseline_metrics[metric]
                current_val = current_metrics[metric]
                
                if baseline_val > 0:
                    relative_change = (current_val - baseline_val) / baseline_val
                    
                    # Critical degradation (>10% drop)
                    if relative_change < -0.10:
                        alert = Alert(
                            title=f"Critical Performance Degradation",
                            message=f"{metric.upper()} dropped by {abs(relative_change)*100:.1f}% (from {baseline_val:.4f} to {current_val:.4f})",
                            severity=AlertSeverity.CRITICAL,
                            metric_name=metric,
                            current_value=current_val,
                            baseline_value=baseline_val,
                            threshold=threshold
                        )
                        alerts.append(alert)
                    
                    # Warning degradation (5-10% drop)
                    elif relative_change < -threshold:
                        alert = Alert(
                            title=f"Performance Degradation Warning",
                            message=f"{metric.upper()} dropped by {abs(relative_change)*100:.1f}% (from {baseline_val:.4f} to {current_val:.4f})",
                            severity=AlertSeverity.WARNING,
                            metric_name=metric,
                            current_value=current_val,
                            baseline_value=baseline_val,
                            threshold=threshold
                        )
                        alerts.append(alert)
        
        return alerts
    
    def check_drift_alerts(
        self,
        drift_results: Dict[str, Any]
    ) -> List[Alert]:
        """
        Check drift detection results and generate alerts.
        
        Args:
            drift_results: Results from drift detection
            
        Returns:
            List of alerts
        """
        alerts = []
        
        if drift_results.get('summary', {}).get('drift_detected', False):
            # Count features with drift
            drift_count = 0
            
            # Check KS test results
            if 'kolmogorov_smirnov' in drift_results:
                ks_results = drift_results['kolmogorov_smirnov']
                drift_count += sum(1 for r in ks_results.values() if r.get('drift_detected', False))
            
            # Check Chi-square results
            if 'chi_square' in drift_results:
                chi2_results = drift_results['chi_square']
                drift_count += sum(1 for r in chi2_results.values() if r.get('drift_detected', False))
            
            # Check PSI results
            if 'psi' in drift_results:
                psi_results = drift_results['psi']
                high_psi_count = sum(1 for psi in psi_results.values() if isinstance(psi, (int, float)) and psi > 0.25)
                drift_count += high_psi_count
            
            if drift_count > 0:
                severity = AlertSeverity.CRITICAL if drift_count > 5 else AlertSeverity.WARNING
                alert = Alert(
                    title=f"Data Drift Detected",
                    message=f"{drift_count} feature(s) show significant drift. Model may need retraining.",
                    severity=severity,
                    metric_name="drift_detection",
                    current_value=drift_count
                )
                alerts.append(alert)
        
        return alerts
    
    def check_fairness_alerts(
        self,
        fairness_report: Dict[str, Any]
    ) -> List[Alert]:
        """
        Check fairness violations and generate alerts.
        
        Args:
            fairness_report: Fairness analysis report
            
        Returns:
            List of alerts
        """
        alerts = []
        
        if fairness_report.get('fairness_violations_detected', False):
            violations = fairness_report.get('violations', [])
            
            for violation in violations:
                metric = violation.get('metric', 'unknown')
                value = violation.get('value', 0)
                threshold = violation.get('threshold', 0)
                feature = violation.get('feature', 'unknown')
                
                # Determine severity based on violation magnitude
                if metric == 'demographic_parity_difference' or metric == 'equal_opportunity_difference':
                    severity = AlertSeverity.CRITICAL if value > 0.2 else AlertSeverity.WARNING
                    alert = Alert(
                        title=f"Fairness Violation: {metric.replace('_', ' ').title()}",
                        message=f"Detected on feature '{feature}': {value:.4f} (threshold: {threshold:.4f})",
                        severity=severity,
                        metric_name=metric,
                        current_value=value,
                        threshold=threshold
                    )
                    alerts.append(alert)
        
        return alerts
    
    def check_latency_alerts(
        self,
        latency_metrics: Dict[str, float],
        max_latency_ms: float = 200.0
    ) -> List[Alert]:
        """
        Check latency metrics and generate alerts.
        
        Args:
            latency_metrics: Latency statistics
            max_latency_ms: Maximum acceptable latency in milliseconds
            
        Returns:
            List of alerts
        """
        alerts = []
        
        p95_latency_ms = latency_metrics.get('p95_latency_seconds', 0) * 1000
        
        if p95_latency_ms > max_latency_ms:
            severity = AlertSeverity.CRITICAL if p95_latency_ms > max_latency_ms * 2 else AlertSeverity.WARNING
            alert = Alert(
                title=f"High Prediction Latency",
                message=f"P95 latency is {p95_latency_ms:.2f}ms (threshold: {max_latency_ms}ms)",
                severity=severity,
                metric_name="latency",
                current_value=p95_latency_ms,
                threshold=max_latency_ms
            )
            alerts.append(alert)
        
        return alerts
    
    def get_alerts_summary(self) -> Dict[str, int]:
        """Get summary of alerts by severity."""
        summary = {
            'critical': 0,
            'warning': 0,
            'info': 0,
            'total': len(self.alerts)
        }
        
        for alert in self.alerts:
            if alert.severity == AlertSeverity.CRITICAL:
                summary['critical'] += 1
            elif alert.severity == AlertSeverity.WARNING:
                summary['warning'] += 1
            else:
                summary['info'] += 1
        
        return summary


__all__ = [
    "Alert",
    "AlertSeverity",
    "AlertManager"
]

