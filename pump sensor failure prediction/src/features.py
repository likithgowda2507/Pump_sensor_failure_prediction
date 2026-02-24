import pandas as pd
import numpy as np
from src.utils import logger

def create_rolling_features(df, sensor_cols, windows=[3, 6]):
    logger.info(f"Creating rolling features for windows {windows}")
    for col in sensor_cols:
        for window in windows:
            # Rolling Mean: Captures average sensor behavior over time, smoothing out noise
            df[f'{col}_roll_mean_{window}h'] = df.groupby('Pump_ID')[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
            # Rolling Std: Captures volatility/instability which often precedes failure
            if window >= 3:
                df[f'{col}_roll_std_{window}h'] = df.groupby('Pump_ID')[col].transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
    return df

def create_lag_features(df, sensor_cols, lags=[1, 3, 6]):
    logger.info(f"Creating lag features for lags {lags}")
    for col in sensor_cols:
        for lag in lags:
            # Lag Features: Allow the model to see historical trends and shifts
            df[f'{col}_lag_{lag}h'] = df.groupby('Pump_ID')[col].shift(lag).bfill()
    return df

def create_domain_features(df):
    logger.info("Creating domain-specific features")
    # Days since Last Maintenance: Machines degrade over time since last service
    if 'Last_Maintenance_Date' in df.columns:
        df['days_since_maint'] = (df['Timestamp'] - pd.to_datetime(df['Last_Maintenance_Date'])).dt.days.fillna(0)
    
    # Rate of Change (for Vibration): Sudden spikes in vibration are key indicators of mechanical wear
    if 'Vibration' in df.columns:
        df['vibration_roc'] = df.groupby('Pump_ID')['Vibration'].diff().fillna(0)
    elif 'Vibration_mm_s' in df.columns:
        df['vibration_roc'] = df.groupby('Pump_ID')['Vibration_mm_s'].diff().fillna(0)
    
    # Cumulative Runtime
    df['total_runtime'] = df.groupby('Pump_ID')['Timestamp'].cumcount()
    
    return df

def feature_engineering_pipeline(df):
    # Identify sensor columns for rolling/lag
    sensor_cols = ['Vibration', 'Bearing_Temperature', 'Inlet_Pressure', 'Outlet_Pressure', 'Flow_Rate', 'Current', 'RPM']
    sensor_cols = [c for c in sensor_cols if c in df.columns]
    
    df = create_rolling_features(df, sensor_cols)
    df = create_lag_features(df, sensor_cols)
    df = create_domain_features(df)
    
    # Drop columns that could lead to data leakage or are redundant timestamps
    leakage_candidates = ['Failure_event', 'failure_event', 'Last_Maintenance_Date'] 
    df = df.drop(columns=[c for c in leakage_candidates if c in df.columns])
    
    return df
