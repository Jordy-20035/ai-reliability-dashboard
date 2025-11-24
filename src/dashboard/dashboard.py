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

# Custom CSS - White background with blue accents
st.markdown("""
<style>
    /* Main background - White */
    .main {
        background-color: #ffffff;
    }
    .stApp {
        background-color: #ffffff;
    }
    
    /* Header styling - Blue accent */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0066cc;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1.75rem;
        font-weight: 600;
        color: #003366;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #0066cc;
        padding-bottom: 0.5rem;
    }
    
    /* Sidebar styling - Modern nav */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0066cc 0%, #004499 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        padding-top: 3rem;
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    /* Radio buttons styling for nav */
    [data-testid="stSidebar"] label {
        color: #ffffff;
        font-weight: 500;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin: 0.25rem 0;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] label:hover {
        background-color: rgba(255, 255, 255, 0.1);
        cursor: pointer;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] label[aria-checked="true"] {
        background-color: rgba(255, 255, 255, 0.2);
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Metric cards */
    .metric-card {
        padding: 1.5rem;
        border-radius: 8px;
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Alerts styling */
    .alert-critical {
        padding: 1rem 1.25rem;
        border-radius: 8px;
        background-color: #fff5f5;
        border-left: 4px solid #dc3545;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(220, 53, 69, 0.1);
    }
    
    .alert-warning {
        padding: 1rem 1.25rem;
        border-radius: 8px;
        background-color: #fffbf0;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(255, 193, 7, 0.1);
    }
    
    .alert-info {
        padding: 1rem 1.25rem;
        border-radius: 8px;
        background-color: #f0f7ff;
        border-left: 4px solid #0066cc;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 102, 204, 0.1);
    }
    
    .alert-success {
        padding: 1rem 1.25rem;
        border-radius: 8px;
        background-color: #f0fff4;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(40, 167, 69, 0.1);
    }
    
    /* Buttons - Blue accent */
    .stButton > button {
        background-color: #0066cc;
        color: white;
        border-radius: 6px;
        border: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #0052a3;
        box-shadow: 0 4px 8px rgba(0, 102, 204, 0.2);
    }
    
    /* Selectbox and other inputs */
    .stSelectbox > div > div {
        background-color: white;
        border: 1px solid #cccccc;
        border-radius: 6px;
    }
    
    /* Dataframe styling */
    .dataframe {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #003366;
        font-weight: 600;
    }
    
    [data-testid="stMetricLabel"] {
        color: #666666;
    }
    
    /* Divider */
    hr {
        border-top: 1px solid #e0e0e0;
        margin: 2rem 0;
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


def display_alerts(alert_manager: AlertManager, show_all: bool = False):
    """Display alerts at the top of the page."""
    alerts = alert_manager.get_active_alerts()
    
    if not alerts:
        return
    
    # Filter alerts based on show_all - show only warnings and critical if not show_all
    if not show_all:
        alerts = [a for a in alerts if a.severity in [AlertSeverity.WARNING, AlertSeverity.CRITICAL]]
    
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
    
    # Sidebar Navigation - Modern design
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0 2rem 0;">
            <h1 style="color: #ffffff; font-size: 1.8rem; margin: 0;">🔍 AI Monitor</h1>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem; margin: 0.5rem 0 0 0;">
                Trustworthy AI Dashboard
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        
        # Navigation menu with icons
        nav_options = {
            "Home": "🏠",
            "Data Explorer": "📂",
            "Model Training": "🤖",
            "Performance Monitoring": "📈",
            "Drift Detection": "🌊",
            "Fairness Analysis": "⚖️",
            "Live Predictions": "🔮"
        }
        
        # Create radio buttons with icons
        selected = st.radio(
            "Pages",
            options=list(nav_options.keys()),
            label_visibility="collapsed",
            format_func=lambda x: f"{nav_options[x]} {x}"
        )
        page = selected
    
    # Main content area
    # Header
    st.markdown('<p class="main-header">🔍 Trustworthy AI Monitor</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: #666666; font-size: 1.1rem; margin-bottom: 2rem;">Automated MLOps System for Model Reliability and Fairness</p>', unsafe_allow_html=True)
    
    # Display alerts at the top
    display_alerts(st.session_state.alert_manager, show_all=False)
    
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
    dataset_name = st.selectbox(
        "Select Dataset",
        ["adult", "compas", "synthetic"]
    )
    
    # Show info about COMPAS if selected
    if dataset_name == "compas":
        compas_path = config.data.raw_data_dir / "compas-scores-two-years.csv"
        if not compas_path.exists():
            st.warning(
                f"⚠️ COMPAS dataset not found at: {compas_path}\n\n"
                "Please download from: https://github.com/propublica/compas-analysis\n"
                "Save as: `data/raw/compas-scores-two-years.csv`"
            )
    
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
    
    # Show previously trained metrics if available
    if 'training_metrics' in st.session_state and st.session_state.training_metrics is not None:
        st.info("📊 Showing results from previous training. Train a new model to update.")
        metrics = st.session_state.training_metrics
        
        st.markdown("### Previous Training Results")
        
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
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ROC AUC", f"{metrics['roc_auc']:.4f}")
            with col2:
                if 'pr_auc' in metrics:
                    st.metric("PR AUC", f"{metrics['pr_auc']:.4f}")
        
        # Show confusion matrix
        st.markdown("### Confusion Matrix")
        try:
            if 'confusion_matrix' in metrics and metrics['confusion_matrix'] is not None:
                cm = metrics['confusion_matrix']
                if not isinstance(cm, np.ndarray):
                    cm = np.array(cm)
                fig = plot_confusion_matrix(cm)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### Confusion Matrix Values")
                cm_df = pd.DataFrame(
                    cm,
                    index=['Actual Negative', 'Actual Positive'],
                    columns=['Predicted Negative', 'Predicted Positive']
                )
                st.dataframe(cm_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not display confusion matrix: {e}")
        
        st.markdown("---")
    
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
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("ROC AUC", f"{metrics['roc_auc']:.4f}")
                    with col2:
                        if 'pr_auc' in metrics:
                            st.metric("PR AUC", f"{metrics['pr_auc']:.4f}")
                
                # Confusion matrix
                st.markdown("### Confusion Matrix")
                try:
                    if 'confusion_matrix' in metrics and metrics['confusion_matrix'] is not None:
                        cm = metrics['confusion_matrix']
                        # Ensure it's a numpy array
                        if not isinstance(cm, np.ndarray):
                            cm = np.array(cm)
                        
                        # Debug: Show raw confusion matrix
                        logger.debug(f"Confusion matrix shape: {cm.shape}, values: {cm}")
                        
                        # Ensure 2D array
                        if cm.ndim == 1:
                            # If 1D, reshape to 2D
                            size = int(np.sqrt(len(cm)))
                            cm = cm.reshape(size, size)
                        elif cm.ndim > 2:
                            cm = cm.reshape(cm.shape[-2], cm.shape[-1])
                        
                        # Create and display the plot with proper labels
                        labels = ['Negative', 'Positive']
                        if cm.shape[0] == 2 and cm.shape[1] == 2:
                            fig = plot_confusion_matrix(cm, labels=labels)
                        else:
                            fig = plot_confusion_matrix(cm)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Also show as a table for clarity
                        st.markdown("#### Confusion Matrix Values")
                        if cm.shape[0] == 2 and cm.shape[1] == 2:
                            cm_df = pd.DataFrame(
                                cm,
                                index=['Actual Negative', 'Actual Positive'],
                                columns=['Predicted Negative', 'Predicted Positive']
                            )
                        else:
                            cm_df = pd.DataFrame(cm)
                        st.dataframe(cm_df, use_container_width=True)
                        
                        # Show individual values
                        st.markdown("#### Breakdown")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**True Negatives (TN):** {metrics.get('tn', int(cm[0, 0]))}")
                            st.write(f"**False Positives (FP):** {metrics.get('fp', int(cm[0, 1]))}")
                        with col2:
                            st.write(f"**False Negatives (FN):** {metrics.get('fn', int(cm[1, 0]))}")
                            st.write(f"**True Positives (TP):** {metrics.get('tp', int(cm[1, 1]))}")
                    else:
                        st.warning("⚠️ Confusion matrix not available in metrics. Available keys: " + ", ".join(metrics.keys()))
                except Exception as cm_error:
                    st.error(f"❌ Error displaying confusion matrix: {cm_error}")
                    import traceback
                    with st.expander("Show Error Details"):
                        st.code(traceback.format_exc())
                
            except Exception as e:
                st.error(f"Error training model: {e}")
                import traceback
                st.code(traceback.format_exc())


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
    
    st.markdown("Enter feature values to get a prediction:")
    
    # This is a simplified version - in production, you'd dynamically generate inputs
    st.info("Feature input interface - To be customized based on your dataset")
    
    if st.button("Get Prediction"):
        st.info("Prediction functionality - integrate with your trained model")


if __name__ == "__main__":
    main()

