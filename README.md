# Trustworthy AI Monitor

**Automated MLOps System for Monitoring ML Model Reliability and Fairness**

## 📋 Overview

This project is a comprehensive MLOps monitoring system designed for Master's thesis research, focusing on:
- **Reliability**: Model performance tracking and stability monitoring
- **Fairness**: Bias detection and fairness metrics across demographic groups
- **Data Drift**: Statistical tests for feature and prediction drift
- **Real-time Monitoring**: REST API and dashboard for live metrics

The system is designed to be lightweight, CPU-efficient, and modular, making it accessible for organizations with limited computational resources.

## 🎯 Key Features

- **Multi-dataset Support**: Adult Income (UCI), COMPAS, synthetic datasets
- **Comprehensive Metrics**:
  - Performance: Accuracy, Precision, Recall, F1, AUC-ROC
  - Fairness: Demographic Parity, Equal Opportunity, Disparate Impact
  - Drift Detection: KS Test, PSI, Wasserstein Distance
- **Real-time API**: FastAPI backend for model serving and monitoring
- **Interactive Dashboard**: Web-based visualization of all metrics
- **Energy Efficient**: Optimized for CPU environments
- **Docker Support**: Easy deployment with containerization

## 🏗️ Architecture

```
Data → Preprocessing → Model Training → Monitoring → Dashboard
                                         ↓
                                    Drift Detection
                                    Performance Metrics
                                    Fairness Analysis
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Trustworthy-AI-Monitor

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

### Basic Usage

```python
from src.data.load_data import load_adult_data
from src.models.train_model import train_model
from src.monitoring.performance_metrics import calculate_performance_metrics

# Load data
X_train, X_test, y_train, y_test = load_adult_data()

# Train model
model = train_model(X_train, y_train)

# Monitor performance
metrics = calculate_performance_metrics(model, X_test, y_test)
print(metrics)
```

### Run API Server

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Run Dashboard

```bash
python src/dashboard/dashboard.py
```

## 📊 Datasets

- **Adult Income Dataset**: Predict income level (>50K or <=50K)
- **COMPAS**: Recidivism prediction (requires download)
- **Synthetic Data**: Generated datasets with controlled drift

## 📦 Project Structure

```
Trustworthy-AI-Monitor/
├── data/                  # Datasets (raw, processed, synthetic)
├── notebooks/             # Jupyter notebooks for exploration
├── src/                   # Core Python modules
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # Model training and evaluation
│   ├── monitoring/       # Drift, performance, fairness metrics
│   ├── api/              # FastAPI endpoints
│   ├── utils/            # Helper functions
│   └── dashboard/        # Visualization and dashboard
├── tests/                # Unit and integration tests
├── docker/               # Docker configuration
└── docs/                 # Documentation
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_monitoring.py

# Run with coverage
pytest --cov=src tests/
```

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
cd docker
docker-compose up --build

# API available at: http://localhost:8000
# Dashboard at: http://localhost:8501
```

## 📈 Monitoring Metrics

### Performance Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Confusion Matrix
- Prediction Latency

### Drift Metrics
- Population Stability Index (PSI)
- Kolmogorov-Smirnov Test
- Wasserstein Distance
- Chi-Square Test

### Fairness Metrics
- Demographic Parity Difference
- Equal Opportunity Difference
- Equalized Odds Difference
- Disparate Impact Ratio

## 🔬 Research Focus

This system addresses key challenges in ML operations:
1. **Reliability**: Detecting model degradation over time
2. **Fairness**: Ensuring equitable predictions across groups
3. **Efficiency**: CPU-optimized for resource-constrained environments
4. **Scalability**: Modular design for future SaaS deployment

## 📚 Documentation

- [Research Notes & Planning](NOTES.md) - Research plan, experiments, MVP details

## 🚀 Training Your First Model

To train your first model, simply run:

```bash
python train_first_model.py
```

This will:
1. Load Adult Income dataset (~39K train, ~9K test samples)
2. Preprocess the data (handles numerical and categorical features)
3. Train an XGBoost model
4. Evaluate and save the model
5. Generate visualizations and save metrics

**Example Results:**
- Accuracy: 0.8745
- F1 Score: 0.7089
- ROC AUC: 0.9290

**Outputs:**
- `models/trained_model.pkl` - Trained model
- `models/preprocessor.pkl` - Preprocessing pipeline
- `results/first_model_results.csv` - Performance metrics

**Note:** Core dependencies are installed automatically. For full functionality:
```bash
pip install -r requirements.txt
```

## 🤝 Contributing

This is a research project for Master's thesis. Contributions and suggestions are welcome!

## 📄 License

MIT License - see LICENSE file for details

## 👤 Author

Master's Thesis Project - Automated MLOps for AI Reliability and Fairness

## 🙏 Acknowledgments

- UCI Machine Learning Repository for Adult Income dataset
- ProPublica for COMPAS analysis
- Open-source ML and MLOps community

