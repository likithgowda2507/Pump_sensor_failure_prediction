import pandas as pd
import joblib
import os
from src.features import feature_engineering_pipeline
from src.utils import logger

def predict_maintenance(input_data):
    """
    Accepts a DataFrame of recent sensor readings for a pump asset.
    """
    logger.info("Starting Prediction Sequence")
    
    # Needs historical context for rolling/lag features
    # (In production, this would pull from a live database cache)
    engineered_df = feature_engineering_pipeline(input_data)
    latest_sample = engineered_df.iloc[-1:]
    
    # Load Classification
    clf_model = joblib.load("models/failure_model.pkl")
    clf_scaler = joblib.load("models/clf_scaler.pkl")
    # Identify feature columns (must match training)
    features = joblib.load("models/clf_scaler.pkl").feature_names_in_ # assuming sklearn scaler
    
    X_clf = clf_scaler.transform(latest_sample[features])
    fail_prob = clf_model.predict_proba(X_clf)[0][1] * 100
    
    # Load Regression
    rul_model = joblib.load("models/rul_model.pkl")
    reg_scaler = joblib.load("models/reg_scaler.pkl")
    
    X_reg = reg_scaler.transform(latest_sample[features])
    pred_rul = float(rul_model.predict(X_reg)[0])
    
    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print(f"Failure Probability (Next 6h): {fail_prob:.1f}%")
    print(f"Predicted Remaining Useful Life: {pred_rul:.1f} Hours")
    print("="*50)
    
    return fail_prob, pred_rul

if __name__ == "__main__":
    # Example usage with dummy data (requiring history for features)
    print("Checking for models...")
    if os.path.exists("models/failure_model.pkl"):
        # This script needs a real sample with history to work correctly
        print("Models found. Ready for deployment.")
    else:
        print("Models not found. Run 'python run_pipeline.py' first.")
