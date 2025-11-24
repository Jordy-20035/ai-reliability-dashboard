# Dataset Usage Guide

## Understanding `train_first_model.py`

The `train_first_model.py` script is a **standalone demonstration script** that provides an end-to-end example of the ML pipeline. Here's what it does and when to use it:

### Purpose

1. **Demonstration/Testing**: Shows how to use the codebase modules together
2. **Initial Setup**: Trains a baseline model for monitoring demonstrations
3. **Learning Tool**: Acts as a reference implementation for new users

### What it does:

1. Loads the **Adult Income dataset** (automatically downloads from OpenML if not cached)
2. Preprocesses the data using `DataPreprocessor`
3. Trains an XGBoost model
4. Evaluates the model and saves metrics
5. Saves the trained model and preprocessor to `models/` directory

### When to use it:

- ✅ First time using the project (to get started quickly)
- ✅ Testing that all components work correctly
- ✅ Creating a baseline model for monitoring demonstrations
- ✅ Learning the workflow before using the dashboard

### When NOT to use it:

- ❌ For production workflows (use the dashboard or API instead)
- ❌ For training with COMPAS dataset (it's hardcoded for Adult Income)
- ❌ For custom training configurations

### Important Notes:

- It's **specific to the Adult Income dataset** by design
- The model and preprocessor are saved to `models/trained_model.pkl` and `models/preprocessor.pkl`
- These saved artifacts can then be loaded in the Streamlit dashboard

---

## Adding COMPAS Dataset

The COMPAS dataset is **already supported** in the codebase! Here's how to use it:

### Step 1: Download the Dataset

1. Go to: https://github.com/propublica/compas-analysis
2. Download the file: `compas-scores-two-years.csv`
3. **Place it here**: `data/raw/compas-scores-two-years.csv`

### Step 2: Use in Code

#### Option A: Using the Dashboard

1. Open the Streamlit dashboard
2. Go to "Data Explorer" page
3. The COMPAS dataset should appear in the dropdown once the file is in place
4. Click "Load Dataset"

#### Option B: Using Python Script

```python
from src.data.load_data import load_compas_data

# Load COMPAS dataset
X_train, X_test, y_train, y_test = load_compas_data(
    test_size=0.2,
    random_state=42
)

# Now use it for training, monitoring, etc.
```

#### Option C: Using the Universal Loader

```python
from src.data.load_data import load_dataset

# Load COMPAS dataset
X_train, X_test, y_train, y_test = load_dataset(
    dataset_name="compas",
    test_size=0.2,
    random_state=42
)
```

### Step 3: Train a Model with COMPAS

You can train a model using COMPAS data in several ways:

#### Using the Dashboard:
1. Load COMPAS dataset in "Data Explorer"
2. Go to "Model Training" page
3. Select your model type
4. Click "Train Model"

#### Using Python:
```python
from src.data.load_data import load_compas_data
from src.data.preprocess import DataPreprocessor
from src.models.train_model import ModelTrainer

# Load data
X_train, X_test, y_train, y_test = load_compas_data()

# Preprocess
preprocessor = DataPreprocessor()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Train
trainer = ModelTrainer(model_type='xgboost', random_state=42)
model = trainer.train(X_train_processed, y_train)

# Save
trainer.save_model("models/compas_model.pkl")
preprocessor.save_preprocessor("models/compas_preprocessor.pkl")
```

### COMPAS Dataset Details

- **Target Variable**: `two_year_recid` or `is_recid` (binary classification)
- **Sensitive Features**: `race`, `sex`, `age_cat`
- **Use Case**: Perfect for fairness analysis demonstrations
- **Preprocessing**: The `DataPreprocessor` handles it automatically

### Directory Structure After Adding COMPAS

```
data/
└── raw/
    ├── adult.pkl                    # Cached Adult Income dataset
    └── compas-scores-two-years.csv  # COMPAS dataset (you add this)
```

---

## Summary

| Dataset | How to Get | Where it Goes | Usage |
|---------|-----------|---------------|-------|
| **Adult Income** | Auto-downloaded on first use | `data/raw/adult.pkl` | `train_first_model.py` or dashboard |
| **COMPAS** | Manual download from GitHub | `data/raw/compas-scores-two-years.csv` | Dashboard or Python scripts |
| **Synthetic** | Generated on-the-fly | N/A (not saved) | Dashboard or `generate_synthetic_data()` |

---

## Quick Reference

### To train with Adult Income:
```bash
python train_first_model.py
```

### To use COMPAS in dashboard:
1. Download `compas-scores-two-years.csv` → `data/raw/`
2. Open dashboard → Data Explorer → Select "compas" → Load

### To use COMPAS in Python:
```python
from src.data.load_data import load_dataset
X_train, X_test, y_train, y_test = load_dataset("compas")
```

