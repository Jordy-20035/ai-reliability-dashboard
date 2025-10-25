# Quick Start Guide

Get up and running with Trustworthy AI Monitor in minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Docker and docker-compose for containerized deployment

## Installation

### Option 1: Local Installation

1. **Clone or navigate to the project**
```bash
cd ai-reliability-dashboard
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import src; print('Installation successful!')"
```

### Option 2: Docker Installation

```bash
cd docker
docker-compose up --build
```

Then access:
- API: http://localhost:8000
- Dashboard: http://localhost:8501

## First Steps

### 1. Train Your First Model

Create a simple script `train_first_model.py`:

```python
from src.data.load_data import load_adult_data
from src.data.preprocess import DataPreprocessor
from src.models.train_model import ModelTrainer
from src.models.evaluate_model import ModelEvaluator

# Load data
print("Loading Adult Income dataset...")
X_train, X_test, y_train, y_test = load_adult_data()

# Preprocess
print("Preprocessing data...")
preprocessor = DataPreprocessor()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Train model
print("Training XGBoost model...")
trainer = ModelTrainer(model_type='xgboost')
model = trainer.train(X_train_processed, y_train)

# Evaluate
print("Evaluating model...")
evaluator = ModelEvaluator(model)
metrics = evaluator.evaluate(X_test_processed, y_test)

print(f"\nResults:")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
print(f"ROC AUC: {metrics['roc_auc']:.4f}")

# Save model
print("\nSaving model...")
trainer.save_model('models/my_first_model.pkl')
print("Done! Model saved to models/my_first_model.pkl")
```

Run it:
```bash
python train_first_model.py
```

### 2. Monitor Your Model

Create `monitor_model.py`:

```python
from src.monitoring.drift_metrics import DriftDetector
from src.monitoring.performance_metrics import PerformanceMonitor
from src.data.preprocess import create_drift_data
import joblib

# Load model
model = joblib.load('models/my_first_model.pkl')

# Load test data
from src.data.load_data import load_adult_data
from src.data.preprocess import DataPreprocessor

X_train, X_test, y_train, y_test = load_adult_data()
preprocessor = DataPreprocessor()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Monitor performance
print("Monitoring performance...")
monitor = PerformanceMonitor(model)
results = monitor.monitor(X_test_processed, y_test)
print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
print(f"Mean Latency: {results['latency']['mean_latency_seconds']*1000:.2f}ms")

# Detect drift
print("\nDetecting drift...")
X_drifted = create_drift_data(X_test_processed, drift_intensity=0.3)
detector = DriftDetector(X_test_processed)
drift_results = detector.detect_drift(X_drifted, methods=['ks', 'psi'])
print(f"Drift detected: {drift_results['summary']['drift_detected']}")
```

Run it:
```bash
python monitor_model.py
```

### 3. Launch the Dashboard

```bash
streamlit run src/dashboard/dashboard.py
```

Open http://localhost:8501 in your browser and explore:
- Data Explorer
- Model Training
- Performance Monitoring
- Drift Detection
- Fairness Analysis

### 4. Start the API Server

```bash
uvicorn src.api.main:app --reload
```

Access:
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs

Test with curl:
```bash
# Health check
curl http://localhost:8000/manage/health

# Make a prediction (after loading model)
curl -X POST http://localhost:8000/predict/single \
  -H "Content-Type: application/json" \
  -d '{"features": {"age": 35, "education": "Bachelors", "hours-per-week": 40}}'
```

## Using Jupyter Notebooks

Explore the example notebooks:

```bash
jupyter notebook
```

Then open:
1. `notebooks/01_data_exploration.ipynb` - Explore datasets
2. `notebooks/02_model_training.ipynb` - Train and evaluate models
3. `notebooks/03_monitoring_simulation.ipynb` - Simulate monitoring scenarios

## Common Tasks

### Generate Synthetic Data

```python
from src.data.load_data import generate_synthetic_data

X_train, X_test, y_train, y_test = generate_synthetic_data(
    n_samples=5000,
    n_features=10,
    add_demographics=True  # For fairness testing
)
```

### Compare Multiple Models

```python
from src.models.evaluate_model import compare_models
from src.models.train_model import train_model

# Train multiple models
models = {
    'XGBoost': train_model(X_train, y_train, model_type='xgboost'),
    'Random Forest': train_model(X_train, y_train, model_type='random_forest'),
    'Logistic': train_model(X_train, y_train, model_type='logistic')
}

# Compare
comparison = compare_models(models, X_test, y_test)
print(comparison)
```

### Analyze Fairness

```python
from src.monitoring.fairness_metrics import calculate_fairness_metrics

# Assuming X_test has demographic features
fairness_report = calculate_fairness_metrics(
    X_test, 
    y_test.values, 
    y_pred,
    sensitive_features=['gender', 'race'],
    threshold=0.1
)

print(f"Violations: {fairness_report['total_violations']}")
```

## Configuration

Edit configuration in code:

```python
from src.utils.config import config

# View current config
print(config.model_dump())

# Modify config
config.monitoring.drift_threshold = 0.1
config.api.port = 9000
```

Or create a `config.yaml`:

```yaml
monitoring:
  drift_threshold: 0.1
  fairness_threshold: 0.15
  
model:
  model_type: xgboost
  max_depth: 8
  
api:
  port: 9000
  host: 0.0.0.0
```

## Troubleshooting

### Issue: "No module named 'src'"
**Solution**: Ensure you're running from the project root or add to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: "Model file not found"
**Solution**: Train a model first or check the path:
```python
from pathlib import Path
Path('models').mkdir(exist_ok=True)
```

### Issue: Adult dataset download fails
**Solution**: The first download might take time. If it fails, try:
```python
from src.data.load_data import load_adult_data
X_train, X_test, y_train, y_test = load_adult_data(from_cache=False)
```

### Issue: API won't start
**Solution**: Check if port 8000 is in use:
```bash
# Windows
netstat -ano | findstr :8000

# Linux/Mac
lsof -i :8000
```

## Next Steps

1. **Read the Documentation**
   - [Architecture Overview](docs/architecture.md)
   - [Monitoring Pipeline](docs/monitoring_pipeline.md)

2. **Run Tests**
```bash
pytest tests/ -v
```

3. **Customize for Your Use Case**
   - Add your own datasets in `src/data/load_data.py`
   - Implement custom metrics in `src/monitoring/`
   - Extend the API with new endpoints

4. **Deploy to Production**
   - Review [Docker deployment guide](docker/README.md)
   - Set up monitoring and alerts
   - Configure security settings

## Getting Help

- **Documentation**: Check the `docs/` directory
- **Examples**: See `notebooks/` for detailed examples
- **Issues**: Review error logs in `logs/` directory
- **API Docs**: http://localhost:8000/docs when API is running

## Example End-to-End Workflow

```python
# Complete workflow from data to deployment
from src.data.load_data import load_adult_data
from src.data.preprocess import DataPreprocessor
from src.models.train_model import ModelTrainer
from src.monitoring.performance_metrics import PerformanceMonitor
from src.monitoring.fairness_metrics import FairnessMonitor

# 1. Load and prepare data
X_train, X_test, y_train, y_test = load_adult_data()
preprocessor = DataPreprocessor()
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# 2. Train model
trainer = ModelTrainer(model_type='xgboost')
model = trainer.train(X_train_prep, y_train)

# 3. Evaluate
y_pred = model.predict(X_test_prep)

# 4. Monitor performance
perf_monitor = PerformanceMonitor(model)
perf_results = perf_monitor.monitor(X_test_prep, y_test)
print(f"Performance: {perf_results['metrics']['f1']:.4f}")

# 5. Check fairness (if demographics available)
if 'gender' in X_test.columns:
    fairness = FairnessMonitor(['gender'])
    fairness_report = fairness.comprehensive_fairness_report(
        X_test, y_test.values, y_pred
    )
    print(f"Fairness violations: {fairness_report['total_violations']}")

# 6. Save model
trainer.save_model('models/production_model.pkl')

print("✅ Model ready for deployment!")
```

## Project Structure Quick Reference

```
ai-reliability-dashboard/
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Model training and evaluation
│   ├── monitoring/     # Drift, performance, fairness
│   ├── api/            # FastAPI endpoints
│   ├── dashboard/      # Streamlit dashboard
│   └── utils/          # Logger, config
├── tests/              # Test suite
├── notebooks/          # Jupyter notebooks
├── data/               # Data storage
├── models/             # Model storage
├── docker/             # Docker configuration
└── docs/               # Documentation
```

Happy monitoring! 🔍✨

