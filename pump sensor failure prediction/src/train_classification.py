import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from src.utils import logger, save_model, save_metrics
import joblib

def train_classification(X_train, y_train, X_test, y_test):
    logger.info("Starting Classification Model Training")
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    save_model(scaler, 'clf_scaler')
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(scale_pos_weight=10, random_state=42) # Assuming class imbalance
    }
    
    results = []
    best_model = None
    best_recall = 0
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        
        logger.info(f"{name} Results - Recall: {recall:.4f}, AUC: {auc:.4f}")
        
        results.append({
            "Model": name,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": auc
        })
        
        # Priority on Recall for Failure Prediction
        # Prefer XGBoost if metrics are similar for explainability
        is_best = (recall > best_recall) or (recall == best_recall and name == "XGBoost")
        
        if is_best:
            best_recall = recall
            best_model = model
            best_model_name = name

    results_df = pd.DataFrame(results).set_index("Model")
    print("\n--- Classification Model Comparison ---")
    print(results_df)
    save_metrics(results_df, "classification")
    
    logger.info(f"Best Classification Model: {best_model_name} with Recall: {best_recall:.4f}")
    save_model(best_model, "failure_model")
    
    return best_model, scaler
