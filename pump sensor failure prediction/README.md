# Pump Predictive Maintenance System ⚙️

An End-to-End Machine Learning project predicting pump failures 6-hours in advance and estimating Remaining Useful Life (RUL).

## Features
- **Synthetic Time-Series Generation**: Creates realistic data with gradual degradation curves and rare failures.
- **Time-based Cross-Validation**: Splits data chronologically to avoid data leakage.
- **Feature Engineering**: Implements Lag, Rolling Windows, and a Degradation index.
- **Dual Model Approach**:
  - **Classification**: predicts impending failure in the next 6 hours.
  - **Regression**: estimates the precise Remaining Useful Life (RUL) in hours.
- **SHAP Feature Importance**: Generates explainability plots.
- **Interactive Streamlit Dashboard**: Acts as a Digital Twin overview.

## How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Synthetic Dataset**:
   This simulates pump degradation over time.
   ```bash
   python src/data_generator.py
   ```

3. **Train Models**:
   This runs cleaning, feature engineering, evaluates models, and saves the best ones to `models/`.
   ```bash
   python src/train_models.py
   ```

4. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```

5. *(Optional)* **Run Explainability**:
   ```bash
   python src/explain.py
   ```
