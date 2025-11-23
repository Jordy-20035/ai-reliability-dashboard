# Research Notes & Project Planning

## Research Plan - 6 Iterations

### Iteration 1: Basic Experiments with Data and Models (2-3 weeks)
- Load Adult Income dataset
- Preprocess data
- Train 2-3 baseline models (XGBoost, Random Forest)
- Get initial performance metrics
- **Output**: Baseline model with performance metrics

### Iteration 2: Drift Detection Implementation (2-3 weeks)
- Implement drift detection methods (PSI, KS-test, Chi-square)
- Test on synthetic drift data
- Compare methods at different drift intensities (0.1, 0.3, 0.5)
- **Output**: Drift detection results comparison

### Iteration 3: Performance Monitoring (2-3 weeks)
- Implement performance monitoring
- Simulate gradual degradation
- Track metrics over time
- **Output**: Degradation detection results

### Iteration 4: Fairness Analysis (2-3 weeks)
- Implement fairness metrics (Demographic Parity, Equal Opportunity)
- Analyze on Adult Income dataset (gender if available)
- Group-wise performance metrics
- **Output**: Fairness analysis report

### Iteration 5: Integration & End-to-end Testing (2-3 weeks)
- Integrate all components
- Test full pipeline: data → model → monitoring
- Load testing API
- **Output**: Working integrated system

### Iteration 6: Final Validation (1-2 weeks)
- Compare with existing solutions
- Final experiments
- Prepare presentation
- **Output**: Final report ready for defense

## Key Experiments (Minimum for Thesis)

1. ⭐ **Experiment 1**: Train and compare models (Iteration 1)
2. ⭐ **Experiment 2**: Drift detection comparison (Iteration 2)
3. ⭐ **Experiment 3**: Performance degradation simulation (Iteration 3)
4. ⭐ **Experiment 4**: Fairness analysis on Adult Income (Iteration 4)
5. ⭐ **Experiment 5**: End-to-end system test (Iteration 5)

## MVP (Minimum Viable Product)

**MVP Components (All Already Implemented!):**
- ✅ Data loading & preprocessing
- ✅ Model training (XGBoost, Random Forest)
- ✅ Performance monitoring
- ✅ Drift detection (PSI, KS-test)
- ✅ Fairness metrics (Demographic Parity, Equal Opportunity)
- ✅ FastAPI backend
- ✅ Streamlit dashboard

**For MVP Demo (20 minutes):**
1. Load data (3 min)
2. Train model (4 min)
3. Show API (3 min)
4. Detect drift (4 min)
5. Show fairness analysis (4 min)
6. Dashboard overview (2 min)

## Experiment Protocol Template

For each experiment, document:
1. **Goal**: What to prove/test
2. **Hypothesis**: Clear hypothesis
3. **Methodology**: Step-by-step procedure
4. **Results**: Metrics, visualizations
5. **Conclusions**: Findings, limitations

## Next Steps

### This Week:
- [ ] Load Adult Income dataset
- [ ] Preprocess data
- [ ] Train first model (XGBoost)
- [ ] Get initial metrics
- [ ] Document results

### Next 2 Weeks:
- [ ] Train 2-3 models for comparison
- [ ] Test drift detection methods
- [ ] Document all experiments

