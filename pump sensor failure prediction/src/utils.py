import os
import joblib
import logging
from datetime import datetime

# Setup logging
def setup_logging():
    os.makedirs('outputs', exist_ok=True)
    log_file = f"outputs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def save_model(model, name):
    os.makedirs('models', exist_ok=True)
    path = f"models/{name}.pkl"
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")

def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    logger.error(f"Model at {path} not found.")
    return None

def save_metrics(df, name):
    os.makedirs('outputs', exist_ok=True)
    path = f"outputs/{name}_metrics.csv"
    df.to_csv(path, index=True)
    logger.info(f"Metrics saved to {path}")
