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
TRAINING COMPLETE!
============================================================
Model Performance:
  Accuracy:  0.8745
  F1 Score:  0.7089
  ROC AUC:   0.9290

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

### Option 2: FastAPI Backend

```bash
# Install FastAPI dependencies
pip install fastapi uvicorn pydantic

# Run API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Access:** 
- API: `http://localhost:8000`
- Interactive Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/manage/health`

**Test with curl:**
```bash
# Health check
curl http://localhost:8000/manage/health

# Get model info (after training)
curl http://localhost:8000/manage/info
```

**Make a prediction** (requires trained model):
```bash
curl -X POST http://localhost:8000/predict/single \
  -H "Content-Type: application/json" \
  -d "{\"features\": {\"age\": 35, \"education\": \"Bachelors\", \"hours-per-week\": 40}}"
```

### Option 3: Python Scripts (For Development)

**Train and evaluate a model:**
```python
from src.data.load_data import load_adult_data
from src.data.preprocess import DataPreprocessor
from src.models.train_model import ModelTrainer
from src.models.evaluate_model import ModelEvaluator

# Load data
X_train, X_test, y_train, y_test = load_adult_data()

# Preprocess
preprocessor = DataPreprocessor()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Train
trainer = ModelTrainer(model_type='xgboost')
model = trainer.train(X_train_processed, y_train)

# Evaluate
evaluator = ModelEvaluator(model)
metrics = evaluator.evaluate(X_test_processed, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
```

**Monitor performance with alerts:**
```python
from src.monitoring.performance_metrics import PerformanceMonitor
from src.monitoring.alerts import AlertManager

# Set baseline metrics
baseline_metrics = {
    'accuracy': 0.87,
    'f1': 0.71,
    'roc_auc': 0.93
}

# Monitor with alerts
monitor = PerformanceMonitor(model, baseline_metrics=baseline_metrics, threshold=0.05)
alert_manager = AlertManager()

results = monitor.monitor(X_test_processed, y_test)
current_metrics = results['metrics']

# Generate alerts
alerts = alert_manager.check_performance_degradation(
    current_metrics, baseline_metrics, threshold=0.05
)

# Display alerts
for alert in alerts:
    print(f"[{alert.severity.value.upper()}] {alert.title}: {alert.message}")
```

**Detect drift:**
```python
from src.monitoring.drift_metrics import DriftDetector
from src.data.preprocess import create_drift_data

# Create drifted data
X_drifted = create_drift_data(X_test_processed, drift_intensity=0.3)

# Detect drift
detector = DriftDetector(X_test_processed, threshold=0.05)
drift_results = detector.detect_drift(X_drifted, methods=['ks', 'psi'])

if drift_results['summary']['drift_detected']:
    print("⚠️ Drift detected!")
else:
    print("✅ No drift detected")
```

---

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

## 🧪 Testing the Code Right Now

### Quick Test (5 minutes)

1. **Train a model:**
   ```bash
   python train_first_model.py
   ```
   ✅ You should see training progress and final metrics

2. **Check generated files:**
   ```bash
   # Check model was saved
   dir models\trained_model.pkl
   
   # Check results
   dir results\first_model_results.csv
   ```
   ✅ Files should exist

3. **Run dashboard:**
   ```bash
   streamlit run src/dashboard/dashboard.py
   ```
   ✅ Browser should open to `http://localhost:8501`

4. **In the dashboard:**
   - Click "Home" - should show system status
   - Click "Performance Monitoring" - should show alert system
   - Load data, train model, check alerts

### Comprehensive Test (15 minutes)

1. **Train model:**
   ```bash
   python train_first_model.py
   ```

2. **Test API:**
   ```bash
   # Terminal 1: Start API
   uvicorn src.api.main:app --reload
   
   # Terminal 2: Test health check
   curl http://localhost:8000/manage/health
   ```

3. **Test Dashboard with Alerts:**
   ```bash
   streamlit run src/dashboard/dashboard.py
   ```
   
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

- **Adult Income (UCI)**: Default dataset, ~48K records, income prediction (>50K)
- **COMPAS**: Requires manual download from ProPublica
- **Synthetic Data**: Generated with `generate_synthetic_data()` for testing

---

## 🔧 Requirements

**Minimum (for training only):**
- Python 3.8+
- pandas, numpy, scikit-learn, xgboost, joblib, openml

**Full functionality:**
- See `requirements.txt` for complete list
- FastAPI, Streamlit, Plotly for dashboard
- Additional monitoring libraries

---

## 🚨 Common Issues & Solutions

**Issue: "ModuleNotFoundError: No module named 'xgboost'"**
```bash
pip install xgboost
```

**Issue: "ModuleNotFoundError: No module named 'openml'"**
```bash
pip install openml
```

**Issue: Dashboard won't start**
```bash
pip install streamlit
streamlit run src/dashboard/dashboard.py
```

**Issue: API won't start**
```bash
pip install fastapi uvicorn
uvicorn src.api.main:app --reload
```

**Issue: No alerts showing**
- Make sure you've set a baseline in Performance Monitoring
- Generate some drift (drift intensity > 0.3) to see drift alerts

---

## 📚 Additional Resources

- **Research Notes**: See [NOTES.md](NOTES.md) for research plan and experiment details
- **API Documentation**: Available at `http://localhost:8000/docs` when API is running
- **Logs**: Check `logs/app.log` for detailed logs

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
