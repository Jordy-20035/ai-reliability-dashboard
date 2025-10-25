"""
Streamlit dashboard for monitoring visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime

from src.data.load_data import load_dataset, load_processed_data
from src.data.preprocess import DataPreprocessor, create_drift_data
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model
from src.monitoring.drift_metrics import DriftDetector, detect_prediction_drift
from src.monitoring.performance_metrics import PerformanceMonitor
from src.monitoring.fairness_metrics import FairnessMonitor
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


def main():
    """Main dashboard application."""
    
    # Initialize session state
    load_model_and_data()
    
    # Header
    st.markdown('<p class="main-header">🔍 Trustworthy AI Monitor</p>', unsafe_allow_html=True)
    st.markdown("**Automated MLOps System for Model Reliability and Fairness**")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Home", "Data Explorer", "Model Training", "Performance Monitoring", 
         "Drift Detection", "Fairness Analysis", "Live Predictions"]
    )
    
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
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_status = "✅ Loaded" if st.session_state.model is not None else "❌ Not Loaded"
        st.metric("Model", model_status)
    
    with col2:
        data_status = "✅ Loaded" if st.session_state.X_train is not None else "❌ Not Loaded"
        st.metric("Data", data_status)
    
    with col3:
        st.metric("Version", "0.1.0")


def show_data_explorer():
    """Data exploration page."""
    st.markdown('<p class="sub-header">📂 Data Explorer</p>', unsafe_allow_html=True)
    
    # Dataset selection
    dataset_name = st.selectbox(
        "Select Dataset",
        ["adult", "synthetic"]
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
                
                # Confusion matrix
                st.markdown("### Confusion Matrix")
                fig = plot_confusion_matrix(metrics['confusion_matrix'])
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error training model: {e}")


def show_performance_monitoring():
    """Performance monitoring page."""
    st.markdown('<p class="sub-header">📈 Performance Monitoring</p>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the Model Training page.")
        return
    
    if st.button("Calculate Performance Metrics"):
        with st.spinner("Calculating metrics..."):
            try:
                X_test = st.session_state.X_test
                if st.session_state.preprocessor is not None:
                    X_test = st.session_state.preprocessor.transform(X_test)
                
                monitor = PerformanceMonitor(st.session_state.model)
                results = monitor.monitor(X_test, st.session_state.y_test)
                
                # Display metrics
                st.markdown("### Performance Metrics")
                
                metrics = results['metrics']
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.4f}")
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.4f}")
                with col4:
                    st.metric("F1 Score", f"{metrics['f1']:.4f}")
                
                # Latency metrics
                st.markdown("### Latency Metrics")
                
                latency = results['latency']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Mean Latency", f"{latency['mean_latency_seconds']*1000:.2f}ms")
                with col2:
                    st.metric("P95 Latency", f"{latency['p95_latency_seconds']*1000:.2f}ms")
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
                
                st.markdown("### Drift Detection Results")
                
                if drift_results['summary']['drift_detected']:
                    st.warning("⚠️ Drift detected in the data!")
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
                
                # Display results
                st.markdown("### Fairness Report")
                
                if fairness_report['fairness_violations_detected']:
                    st.warning(f"⚠️ {fairness_report['total_violations']} fairness violations detected!")
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

