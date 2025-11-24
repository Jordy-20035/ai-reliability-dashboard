"""
Plotting utilities for dashboard visualizations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_performance_metrics(
    metrics_history: pd.DataFrame,
    metrics_to_plot: List[str] = ['accuracy', 'f1', 'precision', 'recall']
) -> go.Figure:
    """
    Plot performance metrics over time.
    
    Args:
        metrics_history: DataFrame with metrics history
        metrics_to_plot: List of metrics to plot
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    for metric in metrics_to_plot:
        if metric in metrics_history.columns:
            fig.add_trace(go.Scatter(
                x=metrics_history.index if isinstance(metrics_history.index, pd.DatetimeIndex) else list(range(len(metrics_history))),
                y=metrics_history[metric],
                mode='lines+markers',
                name=metric.capitalize(),
                line=dict(width=2),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title="Performance Metrics Over Time",
        xaxis_title="Time" if isinstance(metrics_history.index, pd.DatetimeIndex) else "Iteration",
        yaxis_title="Score",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    labels: List[str] = ['Negative', 'Positive']
) -> go.Figure:
    """
    Plot confusion matrix heatmap with vibrant colors.
    
    Args:
        confusion_matrix: 2D numpy array
        labels: Class labels
        
    Returns:
        Plotly figure
    """
    # Ensure it's a numpy array
    cm = np.array(confusion_matrix).astype(float)
    
    # Ensure it's 2D
    if cm.ndim != 2:
        raise ValueError(f"Confusion matrix must be 2D, got {cm.ndim}D")
    
    # Create labels for axes (use provided labels or default)
    x_labels = labels if len(labels) >= cm.shape[1] else ['Negative', 'Positive'][:cm.shape[1]]
    y_labels = labels if len(labels) >= cm.shape[0] else ['Negative', 'Positive'][:cm.shape[0]]
    
    # Ensure we have exactly the right number of labels
    if len(x_labels) < cm.shape[1]:
        x_labels = x_labels + [f'Class {i}' for i in range(len(x_labels), cm.shape[1])]
    if len(y_labels) < cm.shape[0]:
        y_labels = y_labels + [f'Class {i}' for i in range(len(y_labels), cm.shape[0])]
    
    x_labels = x_labels[:cm.shape[1]]
    y_labels = y_labels[:cm.shape[0]]
    
    # Create text annotations showing the values
    text_array = [[f'{int(val)}' for val in row] for row in cm]
    
    # Use numeric indices for x and y, we'll add labels later
    y_indices = list(range(len(y_labels)))
    x_indices = list(range(len(x_labels)))
    
    # Create the heatmap with vibrant colorscale
    # Custom vibrant blue gradient - very colorful!
    custom_colorscale = [
        [0.0, '#e3f2fd'],    # Very light blue
        [0.2, '#90caf9'],    # Light blue
        [0.4, '#42a5f5'],    # Medium light blue
        [0.6, '#1e88e5'],    # Medium blue
        [0.8, '#1565c0'],    # Dark blue
        [1.0, '#0d47a1']     # Very dark blue
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm.tolist(),  # Convert numpy array to list for Plotly
        x=x_indices,
        y=y_indices,
        text=text_array,
        texttemplate='<b>%{text}</b>',
        textfont=dict(size=24, color='white', family="Arial Black"),
        colorscale=custom_colorscale,  # Very vibrant blue gradient
        showscale=True,
        colorbar=dict(
            title=dict(text="Count", font=dict(size=14, color='#333')),
            titleside="right",
            len=0.7,
            thickness=20,
            tickfont=dict(size=11)
        ),
        hovertemplate='<b>Predicted:</b> %{x}<br><b>Actual:</b> %{y}<br><b>Count:</b> %{z}<extra></extra>',
        xgap=2,
        ygap=2
    ))
    
    # Update layout for better appearance
    fig.update_layout(
        title=dict(
            text="Confusion Matrix",
            font=dict(size=22, color='#003366', family="Arial"),
            x=0.5,
            xanchor='center',
            pad=dict(b=20)
        ),
        xaxis=dict(
            title="Predicted Label",
            titlefont=dict(size=14, color='#666666'),
            tickmode='array',
            tickvals=x_indices,
            ticktext=x_labels,
            tickfont=dict(size=13, color='#333333'),
            side='bottom'
        ),
        yaxis=dict(
            title="True Label",
            titlefont=dict(size=14, color='#666666'),
            tickmode='array',
            tickvals=y_indices,
            ticktext=y_labels,
            tickfont=dict(size=13, color='#333333'),
            autorange='reversed'  # Reverse y-axis so [0,0] is top-left
        ),
        template='plotly_white',
        height=500,
        width=600,
        margin=dict(l=120, r=80, t=100, b=100),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial")
    )
    
    return fig


def plot_drift_scores(
    drift_results: Dict[str, float],
    threshold: float = 0.05
) -> go.Figure:
    """
    Plot drift scores for features.
    
    Args:
        drift_results: Dictionary of {feature: drift_score}
        threshold: Drift threshold line
        
    Returns:
        Plotly figure
    """
    features = list(drift_results.keys())
    scores = list(drift_results.values())
    
    # Color based on threshold
    colors = ['red' if score > threshold else 'green' for score in scores]
    
    fig = go.Figure(data=[
        go.Bar(
            x=features,
            y=scores,
            marker_color=colors,
            text=[f'{score:.4f}' for score in scores],
            textposition='outside'
        )
    ])
    
    # Add threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Threshold ({threshold})",
        annotation_position="right"
    )
    
    fig.update_layout(
        title="Feature Drift Scores",
        xaxis_title="Feature",
        yaxis_title="Drift Score",
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    return fig


def plot_fairness_metrics(
    fairness_results: Dict[str, Dict[str, float]]
) -> go.Figure:
    """
    Plot fairness metrics by sensitive feature.
    
    Args:
        fairness_results: Nested dict of fairness metrics
        
    Returns:
        Plotly figure
    """
    # Extract metrics
    sensitive_features = list(fairness_results.keys())
    
    metrics = ['demographic_parity_difference', 'equal_opportunity_difference', 'disparate_impact_ratio']
    
    fig = make_subplots(
        rows=1, cols=len(metrics),
        subplot_titles=metrics,
        horizontal_spacing=0.1
    )
    
    for i, metric in enumerate(metrics, 1):
        values = []
        for feature in sensitive_features:
            if metric in fairness_results[feature]:
                value = fairness_results[feature][metric]
                if isinstance(value, dict):
                    value = value.get('max_difference', 0)
                values.append(value)
            else:
                values.append(0)
        
        fig.add_trace(
            go.Bar(x=sensitive_features, y=values, name=metric),
            row=1, col=i
        )
    
    fig.update_layout(
        title="Fairness Metrics by Sensitive Feature",
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    return fig


def plot_group_performance(
    group_metrics: Dict[str, Dict[str, float]],
    metric: str = 'f1'
) -> go.Figure:
    """
    Plot performance metrics by group.
    
    Args:
        group_metrics: Dictionary of {group: {metric: value}}
        metric: Metric to plot
        
    Returns:
        Plotly figure
    """
    groups = list(group_metrics.keys())
    values = [group_metrics[g].get(metric, 0) for g in groups]
    counts = [group_metrics[g].get('count', 0) for g in groups]
    
    fig = go.Figure(data=[
        go.Bar(
            x=groups,
            y=values,
            text=[f'{v:.3f}<br>n={c}' for v, c in zip(values, counts)],
            textposition='outside',
            marker_color='lightblue'
        )
    ])
    
    fig.update_layout(
        title=f"{metric.upper()} Score by Group",
        xaxis_title="Group",
        yaxis_title=metric.upper(),
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_prediction_distribution(
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray] = None
) -> go.Figure:
    """
    Plot distribution of predictions.
    
    Args:
        predictions: Array of predictions
        probabilities: Array of prediction probabilities
        
    Returns:
        Plotly figure
    """
    if probabilities is not None:
        # Plot probability distribution
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=probabilities,
            nbinsx=50,
            name='Probability Distribution',
            marker_color='skyblue'
        ))
        
        fig.update_layout(
            title="Prediction Probability Distribution",
            xaxis_title="Predicted Probability",
            yaxis_title="Count",
            template='plotly_white',
            height=400
        )
    else:
        # Plot prediction counts
        pred_counts = pd.Series(predictions).value_counts().sort_index()
        
        fig = go.Figure(data=[
            go.Bar(
                x=pred_counts.index.astype(str),
                y=pred_counts.values,
                marker_color='skyblue'
            )
        ])
        
        fig.update_layout(
            title="Prediction Distribution",
            xaxis_title="Predicted Class",
            yaxis_title="Count",
            template='plotly_white',
            height=400
        )
    
    return fig


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20
) -> go.Figure:
    """
    Plot feature importance.
    
    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        top_n: Number of top features to show
        
    Returns:
        Plotly figure
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return go.Figure()
    
    importances = model.feature_importances_
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    fig = go.Figure(data=[
        go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            marker_color='lightcoral'
        )
    ])
    
    fig.update_layout(
        title=f"Top {top_n} Feature Importances",
        xaxis_title="Importance",
        yaxis_title="Feature",
        template='plotly_white',
        height=max(400, top_n * 20)
    )
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> go.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        
    Returns:
        Plotly figure
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='darkorange', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='navy', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_white',
        height=500
    )
    
    return fig


def create_metrics_summary_table(
    metrics: Dict[str, Any]
) -> pd.DataFrame:
    """
    Create a summary table of metrics.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        DataFrame with formatted metrics
    """
    rows = []
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            rows.append({
                'Metric': key.replace('_', ' ').title(),
                'Value': f'{value:.4f}' if isinstance(value, float) else str(value)
            })
    
    return pd.DataFrame(rows)


__all__ = [
    "plot_performance_metrics",
    "plot_confusion_matrix",
    "plot_drift_scores",
    "plot_fairness_metrics",
    "plot_group_performance",
    "plot_prediction_distribution",
    "plot_feature_importance",
    "plot_roc_curve",
    "create_metrics_summary_table"
]

