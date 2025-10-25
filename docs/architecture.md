# System Architecture

## Overview

The Trustworthy AI Monitor is a modular MLOps system designed for monitoring machine learning model reliability and fairness. The architecture follows a microservices approach with clear separation of concerns.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
├────────────────────┬────────────────────────────────────────────┤
│  Streamlit Dashboard │            API Clients                    │
│  (Visualization)     │        (REST/HTTP Requests)              │
└──────────┬───────────┴────────────────┬───────────────────────────┘
           │                            │
           │                            │
┌──────────▼────────────────────────────▼───────────────────────────┐
│                        FastAPI Backend                             │
├────────────────────────────────────────────────────────────────────┤
│  • Prediction API          • Monitoring API                        │
│  • Management API          • Health Checks                         │
└──────────┬─────────────────────────────────────────────────────────┘
           │
           │
┌──────────▼─────────────────────────────────────────────────────────┐
│                      Core Modules                                   │
├────────────┬───────────────┬──────────────┬─────────────────────────┤
│   Data     │    Models     │  Monitoring  │      Utilities          │
│  Module    │    Module     │   Module     │       Module            │
├────────────┼───────────────┼──────────────┼─────────────────────────┤
│• Load      │• Train        │• Drift       │• Logger                 │
│• Preprocess│• Evaluate     │• Performance │• Config                 │
│• Generate  │• Save/Load    │• Fairness    │                         │
└────────────┴───────────────┴──────────────┴─────────────────────────┘
           │                            │
           │                            │
┌──────────▼────────────────────────────▼───────────────────────────┐
│                      Storage Layer                                 │
├────────────────────────────────────────────────────────────────────┤
│  • File System (Data, Models, Logs)                                │
│  • Optional: PostgreSQL (Metrics, History)                         │
│  • Optional: S3/Cloud Storage (Backups, Archives)                  │
└────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. User Interface Layer

#### Streamlit Dashboard
- **Purpose**: Interactive web interface for monitoring and analysis
- **Features**:
  - Data exploration and visualization
  - Model training interface
  - Real-time performance monitoring
  - Drift detection visualization
  - Fairness analysis reports
- **Technology**: Streamlit, Plotly
- **Port**: 8501

#### REST API Clients
- **Purpose**: Programmatic access to the system
- **Use Cases**: Integration with external systems, automation, batch processing
- **Documentation**: Interactive docs at `/docs` (Swagger/OpenAPI)

### 2. API Layer (FastAPI)

#### Prediction API (`/predict`)
- **Single Prediction**: POST `/predict/single`
- **Batch Prediction**: POST `/predict/batch`
- **Features**:
  - Input validation with Pydantic schemas
  - Automatic preprocessing
  - Probability scoring
  - Response logging

#### Monitoring API (`/monitor`)
- **Performance Metrics**: GET `/monitor/performance`
- **Drift Detection**: GET `/monitor/drift`
- **Fairness Metrics**: GET `/monitor/fairness`
- **Comprehensive Report**: GET `/monitor/report`

#### Management API (`/manage`)
- **Health Check**: GET `/manage/health`
- **Model Info**: GET `/manage/info`
- **Configuration**: GET/PUT `/manage/config`

### 3. Core Modules

#### Data Module (`src/data/`)
- **load_data.py**: 
  - Multi-source data loading (Adult, COMPAS, synthetic)
  - Train/test splitting
  - Data caching
- **preprocess.py**:
  - Feature engineering
  - Numerical/categorical preprocessing
  - Missing value imputation
  - Data drift generation (for testing)

#### Models Module (`src/models/`)
- **train_model.py**:
  - Multiple algorithm support (XGBoost, Random Forest, Logistic Regression, LightGBM)
  - Hyperparameter configuration
  - Cross-validation
  - Model serialization
- **evaluate_model.py**:
  - Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)
  - Confusion matrix
  - ROC/PR curves
  - Model comparison

#### Monitoring Module (`src/monitoring/`)
- **drift_metrics.py**:
  - Kolmogorov-Smirnov test (numerical features)
  - Chi-Square test (categorical features)
  - Population Stability Index (PSI)
  - Wasserstein distance
  - Prediction drift detection
- **performance_metrics.py**:
  - Real-time performance tracking
  - Latency measurement
  - Degradation detection
  - Business metrics calculation
- **fairness_metrics.py**:
  - Demographic parity
  - Equal opportunity
  - Equalized odds
  - Disparate impact ratio
  - Group-wise performance analysis

#### Utilities Module (`src/utils/`)
- **logger.py**: Structured logging with Loguru
- **config.py**: Centralized configuration management with Pydantic

### 4. Storage Layer

#### File System
- **Data**: Raw, processed, and synthetic datasets
- **Models**: Serialized trained models (joblib/pickle)
- **Logs**: Application and error logs

#### Optional: Database (PostgreSQL)
- **Metrics History**: Time-series performance data
- **Alerts**: Triggered alerts and notifications
- **Audit Log**: Model predictions and decisions

## Design Principles

### 1. Modularity
- Each component has a single responsibility
- Clear interfaces between modules
- Easy to extend and modify

### 2. Scalability
- Stateless API design
- Horizontal scaling support
- Asynchronous processing capabilities

### 3. CPU-Optimized
- Lightweight algorithms
- No GPU dependencies
- Efficient memory usage
- Suitable for resource-constrained environments

### 4. Observability
- Comprehensive logging
- Health checks
- Performance metrics
- Error tracking

### 5. Reproducibility
- Random seeds for consistency
- Version control for models and data
- Configuration management

## Data Flow

### Training Pipeline
```
Raw Data → Load → Preprocess → Split → Train → Evaluate → Save Model
                                                    ↓
                                            Performance Baseline
```

### Prediction Pipeline
```
Input → Validation → Preprocessing → Model → Prediction → Monitoring
                                                              ↓
                                                    Drift Detection
                                                    Performance Tracking
                                                    Fairness Analysis
```

### Monitoring Pipeline
```
New Data → Baseline Comparison → Statistical Tests → Alert System
              ↓
      Performance Metrics
      Drift Metrics  
      Fairness Metrics
              ↓
        Dashboard/API
```

## Technology Stack

### Core
- **Python 3.8+**: Main programming language
- **scikit-learn**: Traditional ML algorithms
- **XGBoost/LightGBM**: Gradient boosting
- **pandas/numpy**: Data manipulation

### API & Web
- **FastAPI**: REST API framework
- **Uvicorn**: ASGI server
- **Streamlit**: Dashboard framework
- **Plotly**: Interactive visualizations

### Monitoring
- **scipy**: Statistical tests
- **fairlearn**: Fairness metrics
- **alibi-detect**: Drift detection
- **evidently**: Monitoring reports

### Infrastructure
- **Docker**: Containerization
- **docker-compose**: Multi-container orchestration
- **Loguru**: Logging
- **Pydantic**: Data validation

### Testing
- **pytest**: Testing framework
- **httpx**: Async HTTP client for API testing

## Deployment Options

### 1. Local Development
```bash
# Run API
uvicorn src.api.main:app --reload

# Run Dashboard
streamlit run src/dashboard/dashboard.py
```

### 2. Docker Compose
```bash
docker-compose up
```

### 3. Kubernetes (Future)
- Helm charts for deployment
- Horizontal pod autoscaling
- Persistent volume claims for data

### 4. Cloud Deployment
- **AWS**: ECS/EKS, S3, RDS
- **GCP**: Cloud Run, GCS, Cloud SQL
- **Azure**: AKS, Blob Storage, Azure SQL

## Security Considerations

### Authentication & Authorization
- API key authentication (to be implemented)
- Role-based access control
- Rate limiting

### Data Privacy
- No PII in logs
- Encrypted storage for sensitive data
- GDPR compliance features

### Network Security
- HTTPS/TLS encryption
- CORS configuration
- Input validation and sanitization

## Performance Optimization

### API
- Connection pooling
- Caching frequently accessed data
- Async endpoints for I/O operations
- Request batching

### Model Serving
- Model pre-loading
- Batch prediction optimization
- Feature caching
- JIT compilation where applicable

### Monitoring
- Sampling for large datasets
- Incremental statistics calculation
- Background task processing
- Efficient data structures

## Extensibility

### Adding New Datasets
1. Implement loader in `src/data/load_data.py`
2. Add preprocessing logic if needed
3. Update schemas in `src/api/schemas.py`

### Adding New Models
1. Add model class in `src/models/train_model.py`
2. Implement evaluation metrics
3. Update API endpoints

### Adding New Metrics
1. Implement metric calculation in respective monitoring module
2. Add to API routes
3. Create visualization in dashboard

### Adding New Features
1. Define feature in configuration
2. Implement extraction logic
3. Update preprocessing pipeline

## Future Enhancements

1. **Advanced Monitoring**
   - Automated alert system
   - Anomaly detection
   - Concept drift detection
   - Feature importance tracking

2. **Model Management**
   - Model versioning
   - A/B testing framework
   - Champion/challenger models
   - Automated retraining

3. **Enhanced Fairness**
   - Bias mitigation techniques
   - Causal fairness analysis
   - Intersectional fairness
   - Fairness-aware learning

4. **Production Features**
   - Kubernetes deployment
   - CI/CD pipeline
   - Automated testing
   - Performance benchmarking

5. **User Features**
   - User management
   - Custom dashboards
   - Report scheduling
   - Email notifications

