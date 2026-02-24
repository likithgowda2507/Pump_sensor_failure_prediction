import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from src.utils import logger, save_model, save_metrics

def train_regression(X_train, y_train, X_test, y_test):
    logger.info("Starting Regression Model Training for RUL")
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    save_model(scaler, 'reg_scaler')
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    }
    
    results = []
    best_model = None
    best_rmse = float('inf')
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        logger.info(f"{name} Results - RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2-Score": r2
        })
        
        # Priority on lowest RMSE
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = name

    results_df = pd.DataFrame(results).set_index("Model")
    print("\n--- Regression Model Comparison ---")
    print(results_df)
    save_metrics(results_df, "regression")
    
    logger.info(f"Best Regression Model: {best_model_name} with RMSE: {best_rmse:.4f}")
    save_model(best_model, "rul_model")
    
    return best_model, scaler
