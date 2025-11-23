# Trustworthy AI Monitor

**Automated MLOps System for Monitoring ML Model Reliability and Fairness**

A comprehensive monitoring system for machine learning models focusing on reliability, fairness, and data drift detection. Designed for Master's thesis research with CPU-efficient implementation.

---

## 🎯 Key Features

- **Performance Monitoring**: Track accuracy, precision, recall, F1, ROC-AUC with degradation detection
- **Data Drift Detection**: KS Test, PSI, Chi-Square, Wasserstein Distance
- **Fairness Analysis**: Demographic Parity, Equal Opportunity, Disparate Impact metrics
- **Real-time Alerts**: Visual alerts on dashboard when metrics degrade
- **FastAPI Backend**: REST API for model serving and monitoring
- **Interactive Dashboard**: Streamlit-based web interface with visualizations
- **Multi-dataset Support**: Adult Income (UCI), COMPAS, synthetic datasets

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# Install core dependencies (minimum for training)
pip install pandas numpy scikit-learn xgboost joblib openml

# Or install all dependencies
pip install -r requirements.txt
```

**Note:** For full functionality (dashboard, API), install all dependencies from `requirements.txt`. For just training models, the core packages above are sufficient.

### Step 2: Train Your First Model

```bash
python train_first_model.py
```

**What this does:**
1. Loads Adult Income dataset (~39K train, ~9K test samples)
2. Preprocesses data (handles numerical and categorical features)
3. Trains an XGBoost model
4. Evaluates performance
5. Saves model and results

**Expected output:**
```
Model saved to: models/trained_model.pkl
Results saved to: results/first_model_results.csv
```

**Generated files:**
- `models/trained_model.pkl` - Trained XGBoost model
- `models/preprocessor.pkl` - Preprocessing pipeline
- `results/first_model_results.csv` - Performance metrics

---

## 📊 How to Run and Test

### Option 1: Interactive Dashboard (Recommended for Exploration)

```bash
# Install Streamlit if not already installed
pip install streamlit plotly

# Run the dashboard
streamlit run src/dashboard/dashboard.py
```

**Access:** Open your browser to `http://localhost:8501`

**Dashboard Features:**
- **Home**: System status and alert summary
- **Data Explorer**: Load and explore datasets
- **Model Training**: Train models interactively
- **Performance Monitoring**: Real-time metrics with **alerts** when degradation detected
- **Drift Detection**: Test data drift with visualizations
- **Fairness Analysis**: Analyze model fairness across groups
- **Live Predictions**: Make predictions through the UI

**Testing Performance Monitoring with Alerts:**
1. Load dataset in "Data Explorer"
2. Train a model in "Model Training"
3. Go to "Performance Monitoring"
4. Click "Set Current as Baseline" (establishes baseline metrics)
5. Click "Calculate Performance Metrics" again
6. **Alerts will appear** if metrics degrade >5% (warning) or >10% (critical)

**Testing Drift Detection:**
1. Go to "Drift Detection" page
2. Adjust "Drift Intensity" slider (0.1-1.0)
3. Click "Detect Drift"
4. **Alerts appear** when drift is detected
5. View visualizations showing drift scores


## 📦 Project Structure

```
ai-reliability-dashboard/
├── src/                      # Core source code
│   ├── data/                # Data loading & preprocessing
│   ├── models/              # Model training & evaluation
│   ├── monitoring/          # Monitoring metrics & alerts
│   │   ├── alerts.py        # Alert system ⭐
│   │   ├── drift_metrics.py
│   │   ├── performance_metrics.py
│   │   └── fairness_metrics.py
│   ├── api/                 # FastAPI backend
│   ├── dashboard/           # Streamlit dashboard
│   └── utils/               # Configuration & logging
│
├── models/                   # Trained models
│   ├── trained_model.pkl
│   └── preprocessor.pkl
│
├── data/                     # Datasets
│   ├── raw/                 # Original datasets
│   ├── processed/           # Preprocessed data
│   └── synthetic/           # Synthetic test data
│
├── results/                  # Training results & metrics
│   └── first_model_results.csv
│
├── train_first_model.py     # Training script ⭐
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── NOTES.md                 # Research notes & planning
```

---
   
   **Workflow:**
   - Load Adult Income dataset
   - Train XGBoost model
   - Go to Performance Monitoring
   - Set baseline
   - Calculate metrics again → **See alerts appear** (if any degradation)

4. **Test drift detection:**
   - In dashboard: Go to "Drift Detection"
   - Set drift intensity to 0.5
   - Click "Detect Drift"
   - **Alerts should appear** showing drift detected

---

## 📈 Monitoring Metrics

### Performance Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Confusion Matrix
- Prediction Latency (mean, p95, p99)
- Throughput (predictions/second)

### Drift Detection
- **PSI** (Population Stability Index) - Recommended for continuous monitoring
- **KS Test** (Kolmogorov-Smirnov) - For numerical features
- **Chi-Square Test** - For categorical features
- **Wasserstein Distance** - Distribution distance

### Fairness Metrics
- **Demographic Parity** - Equal positive prediction rates
- **Equal Opportunity** - Equal true positive rates
- **Equalized Odds** - Equal TPR and FPR
- **Disparate Impact Ratio** - Ratio of positive rates between groups

### Alert System
- **Critical Alerts** (🔴): >10% degradation, severe drift, critical fairness violations
- **Warning Alerts** (⚠️): 5-10% degradation, moderate drift, fairness concerns
- **Info Alerts** (ℹ️): General information
- **Success** (✅): All clear

---

## 📊 Datasets

**Adult Income (UCI):** Automatically downloaded on first run → `data/raw/adult.pkl`

**COMPAS:**
- Download: https://github.com/propublica/compas-analysis
- Save to: `data/raw/compas-scores-two-years.csv`

**Synthetic Data:** Generated automatically (no download needed)

---

## 🔧 Requirements

**Minimum (for training only):**
- Python 3.8+
- pandas, numpy, scikit-learn, xgboost, joblib, openml

**Full functionality:**
- See `requirements.txt` for complete list
- FastAPI, Streamlit, Plotly for dashboard
- Additional monitoring libraries


## 📚 Additional Resources

- **Research Notes**: See [NOTES.md](NOTES.md) for research plan and experiment details


---

## 🔬 Research Context

This system was developed for Master's thesis research on:
- Automated MLOps monitoring
- Model reliability tracking
- Fairness in machine learning
- CPU-efficient implementation

---

## 📄 License

MIT License - see LICENSE file for details

---

## 👤 Author

Master's Thesis Project - Automated MLOps for AI Reliability and Fairness
