import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.utils import logger

def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)
    return df

def time_based_split(df, split_ratio=0.8):
    logger.info(f"Performing time-based split at ratio {split_ratio}")
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df

def handle_missing_values(df):
    logger.info("Handling missing values")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
        
    return df

def detect_and_cap_outliers(df, cols, threshold=1.5):
    logger.info(f"Detecting and capping outliers for columns: {cols}")
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    return df

def handle_categorical_encoding(df):
    logger.info("Encoding categorical columns")
    categorical_cols = df.select_dtypes(include=['object']).columns
    # Filter out Pump_ID if it's categorical
    categorical_cols = [c for c in categorical_cols if c != 'Pump_ID']
    
    for col in categorical_cols:
        # Simple Label Encoding for demo purposes
        df[col] = df[col].astype('category').cat.codes
    return df

def preprocess_pipeline(file_path):
    df = load_data(file_path)
    df = handle_missing_values(df)
    
    # Cap outliers for sensor columns
    sensor_cols = ['Vibration', 'Bearing_Temperature', 'Inlet_Pressure', 'Outlet_Pressure', 'Flow_Rate', 'Current', 'RPM']
    # Add any columns that look like sensors
    sensor_cols += [c for c in df.columns if any(x in c.lower() for x in ['rolling', 'lag', 'roc'])]
    df = detect_and_cap_outliers(df, [c for c in sensor_cols if c in df.columns])
    
    df = handle_categorical_encoding(df)
    
    train_df, test_df = time_based_split(df)
    
    return train_df, test_df
