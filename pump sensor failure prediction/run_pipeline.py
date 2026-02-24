import os
import pandas as pd
import numpy as np
from src.preprocess import preprocess_pipeline
from src.features import feature_engineering_pipeline
from src.train_classification import train_classification
from src.train_regression import train_regression
from src.evaluate import plot_feature_importance, run_shap_analysis
from src.utils import logger, save_metrics

def run_full_pipeline(dataset_path):
    logger.info("Starting End-to-End Production ML Pipeline")
    
    # 1. Preprocess & Split
    train_df, test_df = preprocess_pipeline(dataset_path)
    
    # 2. Feature Engineering
    logger.info("Engineering features for training set...")
    train_features_df = feature_engineering_pipeline(train_df)
    logger.info("Engineering features for test set...")
    test_features_df = feature_engineering_pipeline(test_df)
    
    # 3. Define Features and Targets
    target_clf = 'Failure_in_Next_6h'
    target_reg = 'RUL_hours'
    
    # Identify non-feature columns
    drop_cols = ['Timestamp', 'Pump_ID', target_clf, target_reg]
    feature_cols = [c for c in train_features_df.columns if c not in drop_cols]
    
    X_train = train_features_df[feature_cols]
    y_train_clf = train_features_df[target_clf]
    y_train_reg = train_features_df[target_reg]
    
    X_test = test_features_df[feature_cols]
    y_test_clf = test_features_df[target_clf]
    y_test_reg = test_features_df[target_reg]
    
    # 4. Train Classification
    best_clf, clf_scaler = train_classification(X_train, y_train_clf, X_test, y_test_clf)
    
    # 5. Train Regression
    best_reg, reg_scaler = train_regression(X_train, y_train_reg, X_test, y_test_reg)
    
    # 6. Evaluate & Explain (XGBoost usually supports feature_importances_)
    logger.info("Running explainability analysis...")
    top_sensors = plot_feature_importance(best_clf, feature_cols, "Classification")
    
    # Run SHAP on a subsample of test data for speed if needed
    X_test_scaled = clf_scaler.transform(X_test)
    sample_size = min(100, len(X_test_scaled))
    run_shap_analysis(best_clf, X_test_scaled[:sample_size], feature_cols, "Classification")
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print(f"Top 5 Contributing Sensors: {top_sensors}")
    print("="*50)

if __name__ == "__main__":
    # Path to the 25K dataset
    path = "data/Pump_Predictive_Maintenance_Dataset_25K.xlsx"
    if not os.path.exists(path):
        # Fallback to CSV if Excel is missing (user mentioned 25K)
        path = "/Users/likith/Downloads/files/data/Pump_Predictive_Maintenance_Dataset_25K.xlsx"
    
    run_full_pipeline(path)
