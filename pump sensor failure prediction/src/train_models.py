import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt

def evaluate_classification(y_true, y_pred, y_prob):
    """Evaluate classification model with relevant metrics."""
    return {
        'Precision': round(precision_score(y_true, y_pred), 4),
        'Recall': round(recall_score(y_true, y_pred), 4),
        'F1 Score': round(f1_score(y_true, y_pred), 4),
        'ROC-AUC': round(roc_auc_score(y_true, y_prob), 4)
    }

def evaluate_regression(y_true, y_pred):
    """Evaluate regression model with relevant metrics."""
    return {
        'MAE': round(mean_absolute_error(y_true, y_pred), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        'R2': round(r2_score(y_true, y_pred), 4)
    }

if __name__ == "__main__":
    print("="*50)
    print("   Training Models (Classification & Regression)  ")
    print("="*50)

    # 1. Load Data
    print("Loading prepared data...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    # Define features and targets
    drop_cols = ['Timestamp', 'Pump_ID', 'Failure_event', 'Failure_type', 'Maintenance_flag', 'RUL_hours', 'Failure_in_6h']
    feature_cols = [col for col in train_df.columns if col not in drop_cols]
    
    X_train = train_df[feature_cols]
    y_train_clf = train_df['Failure_in_6h']
    y_train_reg = train_df['RUL_hours']

    X_test = test_df[feature_cols]
    y_test_clf = test_df['Failure_in_6h']
    y_test_reg = test_df['RUL_hours']

    print(f"Features used ({len(feature_cols)}): {feature_cols[:5]}...")

    # 2. Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')

    # ==========================================
    # 3. Classifiers: Predict Failure in Next 6H
    # ==========================================
    print("\n--- Training Classification Models ---")
    clf_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest Classifier': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced'),
        'XGBoost Classifier': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    clf_results = {}
    best_clf_name = None
    best_clf_score = 0
    best_clf_model = None

    for name, model in clf_models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train_clf)
        
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = evaluate_classification(y_test_clf, y_pred, y_prob)
        clf_results[name] = metrics
        
        # In Industrial Predictive Maintenance, Recall is most important because
        # missing a failure (False Negative) costs much more than checking a pump (False Positive).
        # We'll use F1 or ROC-AUC for overall balance, but prioritize finding failures.
        if metrics['ROC-AUC'] > best_clf_score:
            best_clf_score = metrics['ROC-AUC']
            best_clf_name = name
            best_clf_model = model

    print(f"\nBest Classifier: {best_clf_name}")
    joblib.dump(best_clf_model, 'models/classifier.pkl')
    
    # Save the selected model name for the dashboard
    with open('models/clf_features.json', 'w') as f:
        json.dump(feature_cols, f)

    # ==========================================
    # 4. Regressors: Predict Remaining Useful Life (RUL)
    # ==========================================
    print("\n--- Training Regression Models ---")
    reg_models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'XGBoost Regressor': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    }

    reg_results = {}
    best_reg_name = None
    best_reg_score = float('inf') # For Reg, lower RMSE/MAE is better
    best_reg_model = None

    for name, model in reg_models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train_reg)
        
        y_pred = model.predict(X_test_scaled)
        
        # Clip negative predictions to 0
        y_pred = np.clip(y_pred, a_min=0, a_max=None)
        
        metrics = evaluate_regression(y_test_reg, y_pred)
        reg_results[name] = metrics
        
        if metrics['RMSE'] < best_reg_score:
            best_reg_score = metrics['RMSE']
            best_reg_name = name
            best_reg_model = model

    print(f"\nBest Regressor: {best_reg_name}")
    joblib.dump(best_reg_model, 'models/regressor.pkl')

    # ==========================================
    # 5. Save Comparison Report
    # ==========================================
    os.makedirs('reports', exist_ok=True)
    report = {
        "Classification Models": clf_results,
        "Regression Models": reg_results,
        "Best Model Selection": {
            "Classifier": best_clf_name,
            "Classifier_Reasoning": "Highest ROC-AUC score, indicating the best capability at distinguishing between failure and non-failure events early.",
            "Regressor": best_reg_name,
            "Regressor_Reasoning": "Lowest RMSE and MAE, meaning its predictions for hours left (RUL) are closest to the actual remaining life."
        }
    }
    
    with open('reports/model_comparison.json', 'w') as f:
        json.dump(report, f, indent=4)
    print("Saved model comparison report to 'reports/model_comparison.json'")

    # ==========================================
    # 6. Feature Importance (SHAP)
    # ==========================================
    print("\n--- Generating Feature Explainability ---")
    
    # Limit sample size for SHAP to avoid extreme slow down
    shap_sample = shap.sample(X_test_scaled, 500)
    
    # Check if best classifier is tree-based (RF or XGB)
    if hasattr(best_clf_model, 'feature_importances_'):
        # Global Feature Importance based on model
        importances = best_clf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nTop 5 sensors contributing to failure (based on Global Importance):")
        top_features = []
        for i in range(5):
            feat = feature_cols[indices[i]]
            print(f"{i+1}. {feat}: {importances[indices[i]]:.4f}")
            top_features.append(feat)
            
        # Save top features for dashboard
        with open('reports/top_features.json', 'w') as f:
            json.dump(top_features, f)
            
        # Generate SHAP values
        print("Calculating SHAP values...")
        explainer = shap.TreeExplainer(best_clf_model)
        shap_values = explainer.shap_values(shap_sample)
        
        # Save SHAP Summary Plot
        plt.figure(figsize=(10, 6))
        
        # Handle XGBoost vs Random Forest shap values shape
        if isinstance(shap_values, list): # Random Forest output shape
            shap.summary_plot(shap_values[1], shap_sample, show=False)
        else: # XGBoost output shape
            shap.summary_plot(shap_values, shap_sample, show=False)
            
        plt.tight_layout()
        plt.savefig('reports/shap_summary.png')
        print("Saved SHAP summary plot to 'reports/shap_summary.png'")
    else:
        print("Best classifier is not tree-based. Skipping tree-specific SHAP.")

    print("\nModel Training and Explainability Complete!")
