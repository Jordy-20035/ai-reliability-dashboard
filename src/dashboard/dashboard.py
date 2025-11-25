"""
Streamlit dashboard for monitoring visualization.
"""

import sys
from pathlib import Path

# Add project root to path so imports work from any directory
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from src.data.load_data import load_dataset, load_processed_data
from src.data.preprocess import DataPreprocessor, create_drift_data
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model
from src.monitoring.drift_metrics import DriftDetector, detect_prediction_drift
from src.monitoring.performance_metrics import PerformanceMonitor
from src.monitoring.fairness_metrics import FairnessMonitor
from src.monitoring.alerts import AlertManager, AlertSeverity
from src.dashboard.plots import (
    plot_performance_metrics, plot_confusion_matrix, plot_drift_scores,
    plot_fairness_metrics, plot_group_performance, plot_prediction_distribution,
    plot_feature_importance, plot_roc_curve, create_metrics_summary_table
)
from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Trustworthy AI Monitor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
    }
    .alert-critical {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fee;
        border-left: 5px solid #dc3545;
        margin-bottom: 1rem;
    }
    .alert-warning {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        margin-bottom: 1rem;
    }
    .alert-info {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        margin-bottom: 1rem;
    }
    .alert-success {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def load_model_and_data():
    """Load model and data with caching."""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'alert_manager' not in st.session_state:
        st.session_state.alert_manager = AlertManager()
    if 'baseline_metrics' not in st.session_state:
        st.session_state.baseline_metrics = None
    if 'training_metrics' not in st.session_state:
        st.session_state.training_metrics = None


def display_alerts(alert_manager: AlertManager, show_all: bool = False):
    """Display alerts at the top of the page."""
    alerts = alert_manager.get_active_alerts()
    
    if not alerts:
        return
    
    # Filter alerts based on show_all
    if not show_all:
        alerts = alert_manager.get_warning_alerts()
    
    if not alerts:
        return
    
    # Sort by severity (critical first)
    severity_order = {AlertSeverity.CRITICAL: 0, AlertSeverity.WARNING: 1, AlertSeverity.INFO: 2, AlertSeverity.SUCCESS: 3}
    alerts.sort(key=lambda x: severity_order.get(x.severity, 99))
    
    # Display alerts
    st.markdown("---")
    st.markdown("### 🚨 Active Alerts")
    
    for alert in alerts:
        severity_class = f"alert-{alert.severity.value}"
        icon_map = {
            AlertSeverity.CRITICAL: "🔴",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.SUCCESS: "✅"
        }
        icon = icon_map.get(alert.severity, "ℹ️")
        
        alert_html = f"""
        <div class="{severity_class}">
            <strong>{icon} {alert.title}</strong><br>
            {alert.message}
        </div>
        """
        st.markdown(alert_html, unsafe_allow_html=True)
    
    st.markdown("---")


def main():
    """Main dashboard application."""
    
    # Initialize session state
    load_model_and_data()
    
    # Header
    st.markdown('<p class="main-header">🔍 Trustworthy AI Monitor</p>', unsafe_allow_html=True)
    st.markdown("**Automated MLOps System for Model Reliability and Fairness**")
    
    # Display alerts at the top
    display_alerts(st.session_state.alert_manager, show_all=False)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Home", "Data Explorer", "Model Training", "Performance Monitoring", 
         "Drift Detection", "Fairness Analysis", "Live Predictions"]
    )
    
    # Alert summary in sidebar
    alert_summary = st.session_state.alert_manager.get_alerts_summary()
    if alert_summary['total'] > 0:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Alert Status")
        if alert_summary['critical'] > 0:
            st.sidebar.error(f"🔴 Critical: {alert_summary['critical']}")
        if alert_summary['warning'] > 0:
            st.sidebar.warning(f"⚠️ Warnings: {alert_summary['warning']}")
        if alert_summary['info'] > 0:
            st.sidebar.info(f"ℹ️ Info: {alert_summary['info']}")
    
    # Main content based on selected page
    if page == "Home":
        show_home_page()
    elif page == "Data Explorer":
        show_data_explorer()
    elif page == "Model Training":
        show_model_training()
    elif page == "Performance Monitoring":
        show_performance_monitoring()
    elif page == "Drift Detection":
        show_drift_detection()
    elif page == "Fairness Analysis":
        show_fairness_analysis()
    elif page == "Live Predictions":
        show_live_predictions()


def show_home_page():
    """Home page with overview."""
    st.markdown('<p class="sub-header">Welcome to Trustworthy AI Monitor</p>', unsafe_allow_html=True)
    
    st.markdown("""
    This dashboard provides comprehensive monitoring of machine learning models focusing on:
    
    - **🎯 Performance Monitoring**: Track accuracy, precision, recall, and other metrics
    - **📊 Drift Detection**: Identify data and prediction drift over time
    - **⚖️ Fairness Analysis**: Ensure equitable predictions across demographic groups
    - **🔮 Live Predictions**: Make real-time predictions with the trained model
    
    ### Quick Start
    1. Navigate to **Data Explorer** to load and explore datasets
    2. Go to **Model Training** to train a new model
    3. Use **Performance Monitoring** to evaluate model performance
    4. Check **Drift Detection** and **Fairness Analysis** for comprehensive monitoring
    """)
    
    # System status
    st.markdown('<p class="sub-header">System Status</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        model_status = "✅ Loaded" if st.session_state.model is not None else "❌ Not Loaded"
        st.metric("Model", model_status)
    
    with col2:
        data_status = "✅ Loaded" if st.session_state.X_train is not None else "❌ Not Loaded"
        st.metric("Data", data_status)
    
    with col3:
        alert_summary = st.session_state.alert_manager.get_alerts_summary()
        if alert_summary['critical'] > 0:
            st.metric("Alerts", f"🔴 {alert_summary['critical']}", delta=f"⚠️ {alert_summary['warning']}")
        elif alert_summary['warning'] > 0:
            st.metric("Alerts", f"⚠️ {alert_summary['warning']}", delta="Active")
        else:
            st.metric("Alerts", "✅ None", delta="All clear")
    
    with col4:
        st.metric("Version", "0.1.0")


def show_data_explorer():
    """Data exploration page."""
    st.markdown('<p class="sub-header">📂 Data Explorer</p>', unsafe_allow_html=True)
    
    # Dataset selection
    available_datasets = ["adult", "compas", "synthetic"]
    dataset_name = st.selectbox(
        "Select Dataset",
        available_datasets
    )
    
    # Show info for COMPAS if selected
    if dataset_name == "compas":
        compas_file = config.data.raw_data_dir / "compas-scores-two-years.csv"
        if not compas_file.exists():
            st.warning(f"⚠️ COMPAS dataset not found at {compas_file}. Please download from: https://github.com/propublica/compas-analysis and place it in `data/raw/` directory.")
    
    if st.button("Load Dataset"):
        with st.spinner("Loading dataset..."):
            try:
                X_train, X_test, y_train, y_test = load_dataset(dataset_name)
                
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                st.success(f"Dataset '{dataset_name}' loaded successfully!")
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
    
    # Display data info
    if st.session_state.X_train is not None:
        st.markdown("### Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Train Samples", len(st.session_state.X_train))
        with col2:
            st.metric("Test Samples", len(st.session_state.X_test))
        with col3:
            st.metric("Features", st.session_state.X_train.shape[1])
        with col4:
            class_balance = st.session_state.y_train.value_counts(normalize=True)
            st.metric("Class Balance", f"{class_balance[1]:.2%}")
        
        # Show data sample
        st.markdown("### Data Sample")
        st.dataframe(st.session_state.X_train.head(10))
        
        # Feature statistics
        st.markdown("### Feature Statistics")
        st.dataframe(st.session_state.X_train.describe())


def show_model_training():
    """Model training page."""
    st.markdown('<p class="sub-header">🤖 Model Training</p>', unsafe_allow_html=True)
    
    if st.session_state.X_train is None:
        st.warning("Please load a dataset first in the Data Explorer page.")
        return
    
    # Model configuration
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Select Model Type",
            ["xgboost", "random_forest", "logistic", "lightgbm"]
        )
    
    with col2:
        preprocess = st.checkbox("Preprocess Data", value=True)
    
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            try:
                # Preprocess data
                X_train = st.session_state.X_train
                X_test = st.session_state.X_test
                
                if preprocess:
                    preprocessor = DataPreprocessor()
                    X_train = preprocessor.fit_transform(X_train)
                    X_test = preprocessor.transform(X_test)
                    st.session_state.preprocessor = preprocessor
                
                # Train model
                model = train_model(X_train, st.session_state.y_train, model_type=model_type)
                st.session_state.model = model
                
                # Evaluate
                metrics = evaluate_model(model, X_test, st.session_state.y_test)
                
                # Store metrics in session state for persistence
                st.session_state.training_metrics = metrics
                
                st.success("Model trained successfully!")
                
                # Display metrics
                st.markdown("### Training Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.4f}")
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.4f}")
                with col4:
                    st.metric("F1 Score", f"{metrics['f1']:.4f}")
                
                if 'roc_auc' in metrics:
                    st.metric("ROC AUC", f"{metrics['roc_auc']:.4f}")
                
                # Confusion matrix - with detailed debugging
                st.markdown("### Confusion Matrix")
                
                # Check if confusion matrix exists and is valid
                if 'confusion_matrix' not in metrics:
                    st.error("❌ Confusion matrix key not found in metrics dictionary!")
                    st.json({k: type(v).__name__ for k, v in metrics.items()})
                elif metrics['confusion_matrix'] is None:
                    st.error("❌ Confusion matrix is None!")
                else:
                    cm = metrics['confusion_matrix']
                    # Debug info
                    with st.expander("Debug: Confusion Matrix Info"):
                        st.write(f"Type: {type(cm)}")
                        st.write(f"Shape: {getattr(cm, 'shape', 'N/A')}")
                        st.write(f"Size: {getattr(cm, 'size', len(cm) if hasattr(cm, '__len__') else 'N/A')}")
                        st.write(f"Value: {cm}")
                        st.write(f"Is numpy array: {isinstance(cm, np.ndarray)}")
                    
                    # Validate and plot
                    try:
                        fig = plot_confusion_matrix(cm)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error plotting confusion matrix: {e}")
                        st.write("Raw confusion matrix:")
                        st.write(cm)
                
            except Exception as e:
                st.error(f"Error training model: {e}")


def show_performance_monitoring():
    """Performance monitoring page."""
    st.markdown('<p class="sub-header">📈 Performance Monitoring</p>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the Model Training page.")
        return
    
    # Set baseline metrics button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Set Current as Baseline"):
            if st.session_state.X_test is not None:
                X_test = st.session_state.X_test
                if st.session_state.preprocessor is not None:
                    X_test = st.session_state.preprocessor.transform(X_test)
                
                monitor = PerformanceMonitor(st.session_state.model)
                baseline_metrics = monitor.calculate_metrics(X_test, st.session_state.y_test)
                st.session_state.baseline_metrics = baseline_metrics
                st.session_state.alert_manager.clear_alerts()  # Clear alerts when setting new baseline
                st.success("Baseline metrics set!")
    
    if st.button("Calculate Performance Metrics"):
        with st.spinner("Calculating metrics..."):
            try:
                X_test = st.session_state.X_test
                if st.session_state.preprocessor is not None:
                    X_test = st.session_state.preprocessor.transform(X_test)
                
                # Get or create baseline
                baseline_metrics = st.session_state.baseline_metrics
                if baseline_metrics is None:
                    st.warning("⚠️ No baseline set. Click 'Set Current as Baseline' first, or current metrics will be used as baseline.")
                    # Use current metrics as baseline if none exists
                    monitor = PerformanceMonitor(st.session_state.model)
                    baseline_metrics = monitor.calculate_metrics(X_test, st.session_state.y_test)
                    st.session_state.baseline_metrics = baseline_metrics
                
                monitor = PerformanceMonitor(
                    st.session_state.model,
                    baseline_metrics=baseline_metrics,
                    threshold=0.05
                )
                results = monitor.monitor(X_test, st.session_state.y_test)
                
                # Check for degradation and generate alerts
                current_metrics = results['metrics']
                alerts = st.session_state.alert_manager.check_performance_degradation(
                    current_metrics, baseline_metrics, threshold=0.05
                )
                
                # Check for latency alerts
                latency_alerts = st.session_state.alert_manager.check_latency_alerts(
                    results['latency'], max_latency_ms=200.0
                )
                alerts.extend(latency_alerts)
                
                # Add alerts to manager
                for alert in alerts:
                    st.session_state.alert_manager.add_alert(alert)
                
                # Display alerts if any
                if alerts:
                    st.markdown("### 🚨 Performance Alerts")
                    for alert in alerts:
                        if alert.severity == AlertSeverity.CRITICAL:
                            st.error(f"🔴 **{alert.title}**: {alert.message}")
                        elif alert.severity == AlertSeverity.WARNING:
                            st.warning(f"⚠️ **{alert.title}**: {alert.message}")
                        else:
                            st.info(f"ℹ️ **{alert.title}**: {alert.message}")
                
                # Display metrics
                st.markdown("### Performance Metrics")
                
                metrics = results['metrics']
                col1, col2, col3, col4 = st.columns(4)
                
                # Compare with baseline
                baseline_acc = baseline_metrics.get('accuracy', 0)
                current_acc = metrics['accuracy']
                acc_delta = current_acc - baseline_acc
                
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.4f}", delta=f"{acc_delta:+.4f}")
                
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.4f}")
                
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.4f}")
                
                with col4:
                    st.metric("F1 Score", f"{metrics['f1']:.4f}")
                
                if 'roc_auc' in metrics:
                    st.metric("ROC AUC", f"{metrics['roc_auc']:.4f}")
                
                # Degradation info
                degradation = results['degradation']
                if degradation.get('degradation_detected', False):
                    st.markdown("### ⚠️ Degradation Analysis")
                    degraded_metrics = [
                        m for m, v in degradation['metrics'].items() 
                        if v.get('degraded', False)
                    ]
                    st.warning(f"**Degradation detected in:** {', '.join(degraded_metrics)}")
                    
                    # Show degradation details
                    for metric_name, metric_data in degradation['metrics'].items():
                        if metric_data.get('degraded', False):
                            with st.expander(f"📉 {metric_name.upper()} Degradation Details"):
                                st.write(f"**Baseline:** {metric_data['baseline']:.4f}")
                                st.write(f"**Current:** {metric_data['current']:.4f}")
                                st.write(f"**Change:** {metric_data['relative_change']*100:+.2f}%")
                
                # Latency metrics
                st.markdown("### Latency Metrics")
                
                latency = results['latency']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    latency_ms = latency['mean_latency_seconds']*1000
                    latency_delta = None
                    if latency_ms > 200:
                        latency_delta = f"⚠️ High"
                    st.metric("Mean Latency", f"{latency_ms:.2f}ms", delta=latency_delta)
                
                with col2:
                    p95_latency_ms = latency['p95_latency_seconds']*1000
                    st.metric("P95 Latency", f"{p95_latency_ms:.2f}ms")
                
                with col3:
                    st.metric("Throughput", f"{latency['throughput_per_second']:.0f}/s")
                
            except Exception as e:
                st.error(f"Error calculating metrics: {e}")


def show_drift_detection():
    """Drift detection page."""
    st.markdown('<p class="sub-header">🌊 Drift Detection</p>', unsafe_allow_html=True)
    
    if st.session_state.X_test is None:
        st.warning("Please load a dataset first in the Data Explorer page.")
        return
    
    drift_intensity = st.slider("Drift Intensity", 0.0, 1.0, 0.3, 0.1)
    
    if st.button("Detect Drift"):
        with st.spinner("Detecting drift..."):
            try:
                # Create drifted data
                X_drifted = create_drift_data(st.session_state.X_test, drift_intensity=drift_intensity)
                
                # Detect drift
                detector = DriftDetector(st.session_state.X_test, threshold=0.05)
                drift_results = detector.detect_drift(X_drifted, methods=['ks', 'psi'])
                
                # Generate alerts for drift
                drift_alerts = st.session_state.alert_manager.check_drift_alerts(drift_results)
                for alert in drift_alerts:
                    st.session_state.alert_manager.add_alert(alert)
                
                st.markdown("### Drift Detection Results")
                
                if drift_results['summary']['drift_detected']:
                    st.warning("⚠️ Drift detected in the data!")
                    
                    # Show alerts
                    if drift_alerts:
                        st.markdown("### 🚨 Drift Alerts")
                        for alert in drift_alerts:
                            if alert.severity == AlertSeverity.CRITICAL:
                                st.error(f"🔴 **{alert.title}**: {alert.message}")
                            else:
                                st.warning(f"⚠️ **{alert.title}**: {alert.message}")
                else:
                    st.success("✅ No significant drift detected.")
                
                # Plot KS test results
                if 'kolmogorov_smirnov' in drift_results:
                    st.markdown("### Kolmogorov-Smirnov Test Results")
                    ks_scores = {k: v['statistic'] for k, v in drift_results['kolmogorov_smirnov'].items()}
                    fig = plot_drift_scores(ks_scores, threshold=0.05)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Plot PSI results
                if 'psi' in drift_results:
                    st.markdown("### Population Stability Index (PSI)")
                    fig = plot_drift_scores(drift_results['psi'], threshold=0.1)
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error detecting drift: {e}")


def show_fairness_analysis():
    """Fairness analysis page."""
    st.markdown('<p class="sub-header">⚖️ Fairness Analysis</p>', unsafe_allow_html=True)
    
    if st.session_state.model is None or st.session_state.X_test is None:
        st.warning("Please load data and train a model first.")
        return
    
    # Check for sensitive features
    X_test = st.session_state.X_test
    potential_sensitive = ['gender', 'race', 'sex', 'age_cat']
    sensitive_features = [f for f in potential_sensitive if f in X_test.columns]
    
    if not sensitive_features:
        st.warning("No sensitive features found in the dataset. Available for synthetic or COMPAS datasets.")
        return
    
    selected_features = st.multiselect("Select Sensitive Features", sensitive_features, default=sensitive_features[:1])
    
    if st.button("Analyze Fairness") and selected_features:
        with st.spinner("Analyzing fairness..."):
            try:
                X_test_eval = X_test.copy()
                if st.session_state.preprocessor is not None:
                    X_test_preprocessed = st.session_state.preprocessor.transform(X_test)
                    y_pred = st.session_state.model.predict(X_test_preprocessed)
                else:
                    y_pred = st.session_state.model.predict(X_test)
                
                # Create fairness monitor
                monitor = FairnessMonitor(selected_features, threshold=0.1)
                fairness_report = monitor.comprehensive_fairness_report(
                    X_test, st.session_state.y_test.values, y_pred
                )
                
                # Generate alerts for fairness violations
                fairness_alerts = st.session_state.alert_manager.check_fairness_alerts(fairness_report)
                for alert in fairness_alerts:
                    st.session_state.alert_manager.add_alert(alert)
                
                # Display results
                st.markdown("### Fairness Report")
                
                if fairness_report['fairness_violations_detected']:
                    st.warning(f"⚠️ {fairness_report['total_violations']} fairness violations detected!")
                    
                    # Show alerts
                    if fairness_alerts:
                        st.markdown("### 🚨 Fairness Alerts")
                        for alert in fairness_alerts:
                            if alert.severity == AlertSeverity.CRITICAL:
                                st.error(f"🔴 **{alert.title}**: {alert.message}")
                            else:
                                st.warning(f"⚠️ **{alert.title}**: {alert.message}")
                else:
                    st.success("✅ No fairness violations detected.")
                
                # Display fairness metrics
                for feature in selected_features:
                    if feature in fairness_report['fairness_metrics']:
                        st.markdown(f"#### {feature.title()}")
                        
                        metrics = fairness_report['fairness_metrics'][feature]
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Demographic Parity Diff", f"{metrics['demographic_parity_difference']:.4f}")
                        with col2:
                            st.metric("Equal Opportunity Diff", f"{metrics['equal_opportunity_difference']:.4f}")
                        with col3:
                            st.metric("Disparate Impact Ratio", f"{metrics['disparate_impact_ratio']:.4f}")
                        
                        # Group performance
                        if feature in fairness_report['group_performance']:
                            st.markdown("##### Performance by Group")
                            group_perf = fairness_report['group_performance'][feature]
                            fig = plot_group_performance(group_perf, metric='f1')
                            st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error analyzing fairness: {e}")


def show_live_predictions():
    """Live predictions page."""
    st.markdown('<p class="sub-header">🔮 Live Predictions</p>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the Model Training page.")
        return
    
    if st.session_state.X_train is None:
        st.warning("Please load a dataset first in the Data Explorer page.")
        return
    
    # Get feature information from loaded dataset
    X_sample = st.session_state.X_train
    numerical_features = X_sample.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_sample.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    st.markdown("Enter feature values to get a prediction:")
    st.markdown("---")
    
    # Create input form
    input_data = {}
    
    # Two columns for layout
    col1, col2 = st.columns(2)
    
    # Numerical features in left column
    with col1:
        if numerical_features:
            st.markdown("### Numerical Features")
            for feature in numerical_features:
                min_val = float(X_sample[feature].min()) if len(X_sample) > 0 else 0
                max_val = float(X_sample[feature].max()) if len(X_sample) > 0 else 100
                mean_val = float(X_sample[feature].mean()) if len(X_sample) > 0 else (min_val + max_val) / 2
                
                if feature in ['age', 'hours-per-week', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss']:
                    # Use appropriate ranges for common features
                    if feature == 'age':
                        input_data[feature] = st.number_input(
                            f"{feature.replace('_', ' ').title()}",
                            min_value=18.0,
                            max_value=100.0,
                            value=float(mean_val),
                            step=1.0,
                            key=f"num_{feature}"
                        )
                    elif feature == 'hours-per-week':
                        input_data[feature] = st.number_input(
                            f"{feature.replace('_', ' ').title()}",
                            min_value=1.0,
                            max_value=100.0,
                            value=float(mean_val),
                            step=1.0,
                            key=f"num_{feature}"
                        )
                    else:
                        input_data[feature] = st.number_input(
                            f"{feature.replace('_', ' ').title()}",
                            min_value=min_val,
                            max_value=max_val,
                            value=float(mean_val),
                            key=f"num_{feature}"
                        )
                else:
                    input_data[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}",
                        min_value=min_val,
                        max_value=max_val,
                        value=float(mean_val),
                        key=f"num_{feature}"
                    )
    
    # Categorical features in right column
    with col2:
        if categorical_features:
            st.markdown("### Categorical Features")
            for feature in categorical_features:
                unique_values = sorted(X_sample[feature].dropna().unique().tolist())
                if len(unique_values) > 0:
                    default_idx = 0
                    if len(unique_values) > 10:
                        # For features with many categories, use text input with suggestions
                        input_data[feature] = st.selectbox(
                            f"{feature.replace('_', ' ').title()}",
                            options=unique_values,
                            index=default_idx,
                            key=f"cat_{feature}"
                        )
                    else:
                        input_data[feature] = st.selectbox(
                            f"{feature.replace('_', ' ').title()}",
                            options=unique_values,
                            index=default_idx,
                            key=f"cat_{feature}"
                        )
    
    st.markdown("---")
    
    # Quick fill buttons
    st.markdown("**Quick Actions:**")
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        use_sample = st.button("📋 Load Sample from Dataset", help="Fill form with a random sample from the dataset")
    
    with col_btn2:
        show_sample_data = st.button("👁️ Show Sample Row", help="Display a sample row without filling the form")
    
    # Handle sample data loading
    if use_sample:
        if len(X_sample) > 0:
            sample_row = X_sample.sample(1, random_state=42).iloc[0]
            for feature in numerical_features:
                st.session_state[f"num_{feature}"] = float(sample_row[feature])
            for feature in categorical_features:
                st.session_state[f"cat_{feature}"] = sample_row[feature]
            st.success("Sample data loaded! Scroll up to see the filled form.")
            st.rerun()
    
    # Show sample data
    if show_sample_data:
        if len(X_sample) > 0:
            sample_row = X_sample.sample(1, random_state=42).iloc[0]
            st.markdown("#### Sample Row from Dataset:")
            st.dataframe(sample_row.to_frame().T)
    
    st.markdown("---")
    
    # Make prediction button
    if st.button("🔮 Get Prediction", type="primary", use_container_width=True):
        with st.spinner("Making prediction..."):
            try:
                # Create DataFrame from input
                input_df = pd.DataFrame([input_data])
                
                # Ensure all columns are present and in correct order
                missing_cols = set(X_sample.columns) - set(input_df.columns)
                if missing_cols:
                    # Fill missing columns with default values
                    for col in missing_cols:
                        if col in numerical_features:
                            input_df[col] = X_sample[col].mean()
                        else:
                            input_df[col] = X_sample[col].mode()[0] if len(X_sample[col].mode()) > 0 else X_sample[col].iloc[0]
                
                # Reorder columns to match training data
                input_df = input_df[X_sample.columns]
                
                # Preprocess input
                if st.session_state.preprocessor is not None:
                    input_processed = st.session_state.preprocessor.transform(input_df)
                else:
                    input_processed = input_df
                
                # Make prediction
                model = st.session_state.model
                prediction = model.predict(input_processed)[0]
                
                # Get prediction probability if available
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(input_processed)[0]
                    prob_class_0 = probabilities[0]
                    prob_class_1 = probabilities[1]
                else:
                    prob_class_0 = None
                    prob_class_1 = None
                
                # Display results
                st.markdown("### 🎯 Prediction Results")
                
                # Determine class labels based on dataset
                if st.session_state.y_train is not None:
                    unique_labels = sorted(st.session_state.y_train.unique())
                    class_names = {0: "Class 0", 1: "Class 1"}
                    if len(unique_labels) == 2:
                        # Try to infer class names from the problem
                        if 'adult' in str(X_sample.columns).lower() or any('income' in str(col).lower() for col in X_sample.columns):
                            class_names = {0: "Income ≤50K", 1: "Income >50K"}
                        elif 'compas' in str(X_sample.columns).lower() or any('recid' in str(col).lower() for col in X_sample.columns):
                            class_names = {0: "No Recidivism", 1: "Recidivism"}
                else:
                    class_names = {0: "Class 0", 1: "Class 1"}
                
                pred_class_name = class_names.get(prediction, f"Class {prediction}")
                
                # Display prediction
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    st.markdown(f"**Predicted Class:**")
                    if prediction == 1:
                        st.success(f"🔵 **{pred_class_name}**")
                    else:
                        st.info(f"⚪ **{pred_class_name}**")
                
                with col_res2:
                    if prob_class_0 is not None and prob_class_1 is not None:
                        st.markdown(f"**Confidence:**")
                        pred_prob = prob_class_1 if prediction == 1 else prob_class_0
                        st.metric("Prediction Probability", f"{pred_prob:.2%}")
                
                # Show probabilities
                if prob_class_0 is not None and prob_class_1 is not None:
                    st.markdown("#### Probability Distribution")
                    
                    prob_df = pd.DataFrame({
                        'Class': [class_names[0], class_names[1]],
                        'Probability': [prob_class_0, prob_class_1]
                    })
                    
                    # Create bar chart
                    import plotly.express as px
                    fig = px.bar(
                        prob_df,
                        x='Class',
                        y='Probability',
                        color='Probability',
                        color_continuous_scale='Blues',
                        text='Probability'
                    )
                    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                    fig.update_layout(
                        title="Prediction Probabilities",
                        xaxis_title="Class",
                        yaxis_title="Probability",
                        yaxis=dict(range=[0, 1]),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show probability values
                    col_prob1, col_prob2 = st.columns(2)
                    with col_prob1:
                        st.metric(class_names[0], f"{prob_class_0:.2%}")
                    with col_prob2:
                        st.metric(class_names[1], f"{prob_class_1:.2%}")
                
                # Show input summary
                with st.expander("📋 View Input Summary"):
                    st.dataframe(input_df.T.rename(columns={0: "Value"}))
                    
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                logger.error(f"Prediction error: {e}", exc_info=True)
                with st.expander("Error Details"):
                    st.exception(e)


if __name__ == "__main__":
    main()

