"""
Train the first model for the Trustworthy AI Monitor project.

This script:
1. Loads Adult Income dataset
2. Preprocesses the data
3. Trains an XGBoost model
4. Evaluates the model
5. Saves the model and results
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.load_data import load_adult_data
from src.data.preprocess import DataPreprocessor
from src.models.train_model import ModelTrainer
from src.models.evaluate_model import ModelEvaluator
from src.utils.logger import logger
from src.utils.config import config

def main():
    """Main training pipeline."""
    
    logger.info("="*60)
    logger.info("Starting First Model Training")
    logger.info("="*60)
    
    # Create necessary directories
    models_dir = Path("models")
    results_dir = Path("results")
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    # Step 1: Load data
    logger.info("\n[Step 1/5] Loading Adult Income dataset...")
    try:
        X_train, X_test, y_train, y_test = load_adult_data(
            test_size=0.2,
            random_state=42,
            from_cache=True
        )
        logger.info(f"✓ Dataset loaded successfully")
        logger.info(f"  - Train set: {len(X_train)} samples, {X_train.shape[1]} features")
        logger.info(f"  - Test set: {len(X_test)} samples")
        logger.info(f"  - Classes: {dict(y_train.value_counts())}")
    except Exception as e:
        logger.error(f"✗ Failed to load dataset: {e}")
        logger.error("Attempting to load without cache...")
        X_train, X_test, y_train, y_test = load_adult_data(
            test_size=0.2,
            random_state=42,
            from_cache=False
        )
        logger.info("✓ Dataset loaded successfully (without cache)")
    
    # Step 2: Preprocess data
    logger.info("\n[Step 2/5] Preprocessing data...")
    try:
        preprocessor = DataPreprocessor()
        logger.info("  - Fitting preprocessor on training data...")
        X_train_processed = preprocessor.fit_transform(X_train)
        logger.info("  - Transforming test data...")
        X_test_processed = preprocessor.transform(X_test)
        
        logger.info(f"✓ Preprocessing completed")
        logger.info(f"  - Train features after preprocessing: {X_train_processed.shape[1]}")
        logger.info(f"  - Test features after preprocessing: {X_test_processed.shape[1]}")
        
        # Save preprocessor
        import joblib
        preprocessor_path = models_dir / "preprocessor.pkl"
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"  - Preprocessor saved to {preprocessor_path}")
        
    except Exception as e:
        logger.error(f"✗ Preprocessing failed: {e}")
        raise
    
    # Step 3: Train model
    logger.info("\n[Step 3/5] Training XGBoost model...")
    try:
        trainer = ModelTrainer(
            model_type='xgboost',
            model_params={
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            },
            random_state=42
        )
        
        logger.info("  - Training model...")
        model = trainer.train(X_train_processed, y_train)
        
        logger.info("✓ Model training completed")
        
        # Save model
        model_path = models_dir / "trained_model.pkl"
        trainer.save_model(model_path)
        logger.info(f"  - Model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"✗ Model training failed: {e}")
        raise
    
    # Step 4: Evaluate model
    logger.info("\n[Step 4/5] Evaluating model...")
    try:
        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(X_test_processed, y_test, return_predictions=True)
        
        logger.info("✓ Model evaluation completed")
        logger.info("\n  Performance Metrics:")
        logger.info(f"    - Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"    - Precision: {metrics['precision']:.4f}")
        logger.info(f"    - Recall:    {metrics['recall']:.4f}")
        logger.info(f"    - F1 Score:  {metrics['f1']:.4f}")
        
        if 'roc_auc' in metrics:
            logger.info(f"    - ROC AUC:   {metrics['roc_auc']:.4f}")
        if 'pr_auc' in metrics:
            logger.info(f"    - PR AUC:    {metrics['pr_auc']:.4f}")
        
        # Confusion Matrix
        cm = metrics['confusion_matrix']
        logger.info("\n  Confusion Matrix:")
        logger.info(f"    TN: {metrics['tn']}, FP: {metrics['fp']}")
        logger.info(f"    FN: {metrics['fn']}, TP: {metrics['tp']}")
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([{
            'model': 'xgboost',
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'roc_auc': metrics.get('roc_auc', np.nan),
            'pr_auc': metrics.get('pr_auc', np.nan),
            'tn': metrics['tn'],
            'fp': metrics['fp'],
            'fn': metrics['fn'],
            'tp': metrics['tp'],
            'timestamp': datetime.now().isoformat()
        }])
        
        results_path = results_dir / "first_model_results.csv"
        metrics_df.to_csv(results_path, index=False)
        logger.info(f"\n  - Metrics saved to {results_path}")
        
        # Create visualizations
        logger.info("\n  - Creating visualizations...")
        plots_dir = results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        evaluator.plot_confusion_matrix(
            X_test_processed, y_test,
            save_path=plots_dir / "confusion_matrix.png"
        )
        
        if hasattr(model, 'predict_proba'):
            evaluator.plot_roc_curve(
                X_test_processed, y_test,
                save_path=plots_dir / "roc_curve.png"
            )
        
        logger.info(f"  - Visualizations saved to {plots_dir}")
        
    except Exception as e:
        logger.error(f"✗ Model evaluation failed: {e}")
        raise
    
    # Step 5: Summary
    logger.info("\n[Step 5/5] Training Summary")
    logger.info("="*60)
    logger.info("✓ All steps completed successfully!")
    logger.info(f"\nOutputs:")
    logger.info(f"  - Model: {model_path}")
    logger.info(f"  - Preprocessor: {preprocessor_path}")
    logger.info(f"  - Metrics: {results_path}")
    logger.info(f"  - Plots: {plots_dir}")
    logger.info("\n" + "="*60)
    
    # Print final metrics
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nModel saved to: {model_path}")
    print(f"Results saved to: {results_path}")
    print("="*60 + "\n")
    
    return model, preprocessor, metrics


if __name__ == "__main__":
    try:
        model, preprocessor, metrics = main()
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n✗ Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

