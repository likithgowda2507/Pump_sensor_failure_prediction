import pandas as pd
import numpy as np
from datetime import timedelta
import os

def load_data(filepath='pump_sensor_data.csv'):
    """Load the generated dataset."""
    df = pd.read_csv(filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

def perform_eda(df):
    """Generate a summary report of the dataset."""
    print("=== Dataset Summary ===")
    print(df.info())
    print("\n=== Missing Values Check ===")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    print("\n=== Summary Statistics ===")
    print(df.describe())
    
    # Analyze failure events
    print("\n=== Failure Events Analysis ===")
    print(df['Failure_type'].value_counts())
    
    # Save a correlation matrix report
    # We select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    os.makedirs('reports', exist_ok=True)
    corr_matrix.to_csv('reports/correlation_matrix.csv')
    print("\nCorrelation matrix saved to reports/correlation_matrix.csv")

def handle_missing_values(df):
    """
    Handle missing values.
    Since this is sensor time-series data, forward-fill (ffill) is standard
    because we assume the last recorded sensor value holds until a new reading
    is available. If the first row is missing, we use bfill to catch it.
    """
    # Group by Pump_ID so we don't bleed latest readings into a different pump
    df_clean = df.groupby('Pump_ID', group_keys=False).apply(lambda group: group.ffill().bfill())
    return df_clean

def treat_outliers(df, columns):
    """
    Detect and trim outliers using the Interquartile Range (IQR) method.
    We trim only extreme outliers (e.g., multiplier of 3.0 instead of 1.5) 
    so we don't remove genuine sensor spikes leading up to failure.
    """
    df_out = df.copy()
    for col in columns:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3.0 * IQR
        upper_bound = Q3 + 3.0 * IQR
        
        # Clip the outliers to the bounds instead of removing the row,
        # which would break the time-series continuity.
        df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
    return df_out

def engineer_features(df, sensor_cols):
    """
    Create rolling features, lag features, and rate of change features.
    These features represent trends and recent history which are key for 
    time-series forecasting of failures.
    """
    # Ensure sorted by time within each pump
    df = df.sort_values(by=['Pump_ID', 'Timestamp']).reset_index(drop=True)
    
    engineered_dfs = []
    
    for pump_id, group in df.groupby('Pump_ID'):
        group = group.copy()
        
        for col in sensor_cols:
            # 1. Rolling Mean (3-hour and 6-hour windows)
            # captures smoothed recent trend
            group[f'{col}_rolling_mean_3h'] = group[col].rolling(window=3, min_periods=1).mean()
            group[f'{col}_rolling_mean_6h'] = group[col].rolling(window=6, min_periods=1).mean()
            
            # 2. Rolling Standard Deviation (3-hour and 6-hour windows)
            # captures recent volatility/vibration changes
            group[f'{col}_rolling_std_3h'] = group[col].rolling(window=3, min_periods=1).std().fillna(0)
            group[f'{col}_rolling_std_6h'] = group[col].rolling(window=6, min_periods=1).std().fillna(0)
            
            # 3. Lag features (t-1, t-3, t-6)
            # captures exactly what the value was X hours ago
            group[f'{col}_lag_1h'] = group[col].shift(1).fillna(method='bfill')
            group[f'{col}_lag_3h'] = group[col].shift(3).fillna(method='bfill')
            group[f'{col}_lag_6h'] = group[col].shift(6).fillna(method='bfill')
            
            # 4. Rate of change (Current - Lag_3) / Lag_3
            # captures how fast a sensor is rising/falling
            safe_denom = group[f'{col}_lag_3h'].replace(0, 1e-5) # avoid div by zero
            group[f'{col}_roc_3h'] = (group[col] - group[f'{col}_lag_3h']) / safe_denom

        engineered_dfs.append(group)
        
    df_final = pd.concat(engineered_dfs, ignore_index=True)
    return df_final

def temporal_split(df, train_ratio=0.8):
    """
    Split the dataset based on time.
    For predictive maintenance, we train on past events and test on future events.
    Random shuffling would cause data leakage (peeking into the future).
    We will split EACH pump chronologically.
    """
    train_dfs = []
    test_dfs = []
    
    for pump_id, group in df.groupby('Pump_ID'):
        group = group.sort_values('Timestamp')
        split_idx = int(len(group) * train_ratio)
        
        train_dfs.append(group.iloc[:split_idx])
        test_dfs.append(group.iloc[split_idx:])
        
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    return train_df, test_df

if __name__ == "__main__":
    print("Starting Data Processing and Feature Engineering...")
    os.makedirs('data', exist_ok=True)
    
    # 1. Load
    df = load_data('pump_sensor_data.csv')
    
    # 2. EDA
    perform_eda(df)
    
    # 3. Handle Missing Values
    print("\nCleaning missing values...")
    df = handle_missing_values(df)
    
    # 4. Outlier Treatment
    print("Treating outliers...")
    sensor_cols = [
        'Temperature_C', 'Vibration_mm_s', 'Pressure_bar', 'Flow_rate_lpm',
        'Motor_current_A', 'Voltage_V', 'Power_kW', 'Efficiency_percent',
        'Noise_level_dB', 'Oil_level_percent'
    ]
    df = treat_outliers(df, sensor_cols)
    
    # 5. Feature Engineering
    print("Engineering features (rolling, lags, rate of change)...")
    df = engineer_features(df, sensor_cols)
    
    # 6. Temporal Split
    print("Performing temporal train/test split (80/20)...")
    train_df, test_df = temporal_split(df)
    
    print(f"Train Shape: {train_df.shape}")
    print(f"Test Shape: {test_df.shape}")
    
    # 7. Save
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    print("Saved processed data to 'data/train.csv' and 'data/test.csv'.")
    print("Feature Engineering completed successfully.")
