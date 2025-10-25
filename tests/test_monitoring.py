"""
Tests for monitoring functionality.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.monitoring.drift_metrics import DriftDetector, detect_prediction_drift
from src.monitoring.performance_metrics import PerformanceMonitor, calculate_performance_metrics
from src.monitoring.fairness_metrics import FairnessMonitor, calculate_fairness_metrics
from src.data.load_data import generate_synthetic_data
from src.data.preprocess import create_drift_data


class TestDriftDetection:
    """Test drift detection functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.uniform(0, 10, 1000),
            'feature3': np.random.choice(['A', 'B', 'C'], 1000)
        })
        return X
    
    def test_drift_detector_initialization(self, sample_data):
        """Test drift detector initialization."""
        detector = DriftDetector(sample_data, threshold=0.05)
        
        assert detector.reference_data is not None
        assert detector.threshold == 0.05
        assert detector.reference_stats is not None
    
    def test_kolmogorov_smirnov_test_no_drift(self, sample_data):
        """Test KS test with no drift."""
        detector = DriftDetector(sample_data, threshold=0.05)
        
        # Test on same data (should not detect drift)
        results = detector.kolmogorov_smirnov_test(sample_data)
        
        assert isinstance(results, dict)
        # Most features should not show drift
        drift_count = sum(1 for r in results.values() if r['drift_detected'])
        assert drift_count <= len(results) * 0.1  # Allow 10% false positives
    
    def test_kolmogorov_smirnov_test_with_drift(self, sample_data):
        """Test KS test with drift."""
        detector = DriftDetector(sample_data, threshold=0.05)
        
        # Create drifted data
        drifted_data = create_drift_data(sample_data, drift_intensity=0.5)
        results = detector.kolmogorov_smirnov_test(drifted_data)
        
        # Should detect drift in at least one feature
        drift_count = sum(1 for r in results.values() if r['drift_detected'])
        assert drift_count > 0
    
    def test_population_stability_index(self, sample_data):
        """Test PSI calculation."""
        detector = DriftDetector(sample_data, threshold=0.05)
        
        # Calculate PSI on slightly drifted data
        drifted_data = create_drift_data(sample_data, drift_intensity=0.2)
        psi = detector.population_stability_index(drifted_data, 'feature1')
        
        assert isinstance(psi, float)
        assert psi >= 0  # PSI is always non-negative
    
    def test_prediction_drift_detection(self):
        """Test prediction drift detection."""
        reference_preds = np.random.binomial(1, 0.5, 1000)
        current_preds = np.random.binomial(1, 0.7, 1000)  # Different distribution
        
        results = detect_prediction_drift(reference_preds, current_preds)
        
        assert isinstance(results, dict)
        assert 'drift_detected' in results
        assert 'p_value' in results
        assert 'statistic' in results


class TestPerformanceMonitoring:
    """Test performance monitoring functionality."""
    
    @pytest.fixture
    def trained_model(self):
        """Create a simple trained model."""
        X_train, X_test, y_train, y_test = generate_synthetic_data(
            n_samples=1000,
            random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        return model, X_test, y_test
    
    def test_performance_monitor_initialization(self, trained_model):
        """Test performance monitor initialization."""
        model, X_test, y_test = trained_model
        
        monitor = PerformanceMonitor(model)
        assert monitor.model is not None
    
    def test_calculate_metrics(self, trained_model):
        """Test metrics calculation."""
        model, X_test, y_test = trained_model
        
        monitor = PerformanceMonitor(model)
        metrics = monitor.calculate_metrics(X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        # Check metric values are in valid range
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1'] <= 1
    
    def test_measure_latency(self, trained_model):
        """Test latency measurement."""
        model, X_test, y_test = trained_model
        
        monitor = PerformanceMonitor(model)
        latency_metrics = monitor.measure_latency(X_test[:100], n_iterations=10)
        
        assert 'mean_latency_seconds' in latency_metrics
        assert 'throughput_per_second' in latency_metrics
        assert latency_metrics['mean_latency_seconds'] > 0
    
    def test_detect_degradation(self, trained_model):
        """Test degradation detection."""
        model, X_test, y_test = trained_model
        
        baseline_metrics = {
            'accuracy': 0.9,
            'f1': 0.85,
            'precision': 0.88,
            'recall': 0.82
        }
        
        monitor = PerformanceMonitor(model, baseline_metrics=baseline_metrics, threshold=0.05)
        current_metrics = monitor.calculate_metrics(X_test, y_test)
        
        degradation_results = monitor.detect_degradation(current_metrics)
        
        assert 'degradation_detected' in degradation_results
        assert isinstance(degradation_results['degradation_detected'], bool)


class TestFairnessMonitoring:
    """Test fairness monitoring functionality."""
    
    @pytest.fixture
    def fairness_data(self):
        """Generate data with sensitive features."""
        X_train, X_test, y_train, y_test = generate_synthetic_data(
            n_samples=1000,
            add_demographics=True,
            random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        return X_test, y_test, y_pred
    
    def test_fairness_monitor_initialization(self):
        """Test fairness monitor initialization."""
        monitor = FairnessMonitor(['gender', 'race'], threshold=0.1)
        
        assert monitor.sensitive_features == ['gender', 'race']
        assert monitor.threshold == 0.1
    
    def test_demographic_parity_difference(self, fairness_data):
        """Test demographic parity calculation."""
        X_test, y_test, y_pred = fairness_data
        
        monitor = FairnessMonitor(['gender'], threshold=0.1)
        dpd = monitor.demographic_parity_difference(y_pred, X_test['gender'])
        
        assert isinstance(dpd, float)
        assert 0 <= dpd <= 1
    
    def test_equal_opportunity_difference(self, fairness_data):
        """Test equal opportunity calculation."""
        X_test, y_test, y_pred = fairness_data
        
        monitor = FairnessMonitor(['gender'], threshold=0.1)
        eod = monitor.equal_opportunity_difference(y_test, y_pred, X_test['gender'])
        
        assert isinstance(eod, float)
        assert 0 <= eod <= 1
    
    def test_comprehensive_fairness_report(self, fairness_data):
        """Test comprehensive fairness report."""
        X_test, y_test, y_pred = fairness_data
        
        monitor = FairnessMonitor(['gender', 'race'], threshold=0.1)
        report = monitor.comprehensive_fairness_report(X_test, y_test, y_pred)
        
        assert 'fairness_metrics' in report
        assert 'group_performance' in report
        assert 'violations' in report
        assert 'fairness_violations_detected' in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

