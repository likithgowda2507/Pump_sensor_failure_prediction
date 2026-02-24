import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from src.utils import logger

def plot_feature_importance(model, features, name):
    logger.info(f"Plotting feature importance for {name}")
    os.makedirs('outputs', exist_ok=True)
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
    else:
        logger.warning(f"Model {name} does not support feature_importances_ or coef_")
        return []

    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title(f"Feature Importance - {name}")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(f"outputs/{name}_feature_importance.png")
    plt.close()
    
    # Identify top sensors
    top_indices = indices[:5]
    top_sensors = [features[i] for i in top_indices]
    logger.info(f"Top 5 Contributors for {name}: {top_sensors}")
    return top_sensors

def run_shap_analysis(model, X_test, feature_names, name):
    logger.info(f"Running SHAP analysis for {name}")
    os.makedirs('outputs', exist_ok=True)
    
    # SHAP explainer selection
    try:
        if "XGB" in str(type(model)) or "RandomForest" in str(type(model)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        elif "Logistic" in str(type(model)) or "Linear" in str(type(model)):
            explainer = shap.LinearExplainer(model, X_test)
            shap_values = explainer.shap_values(X_test)
        else:
            explainer = shap.Explainer(model, X_test)
            shap_values = explainer(X_test)
            
        # Summary Plot
        plt.figure()
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(f"outputs/{name}_shap_summary.png")
        plt.close()
        
        logger.info(f"SHAP summary plot saved for {name}")
    except Exception as e:
        logger.error(f"SHAP analysis failed for {name}: {e}")
