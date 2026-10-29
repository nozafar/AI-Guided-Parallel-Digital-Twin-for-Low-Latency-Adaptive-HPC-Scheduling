import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import joblib

class WorkloadPredictor:
    def __init__(self, model_path="models/xgboost_predictor.joblib"):
        self.model_path = model_path
        self.model = None
        
        if os.path.exists(self.model_path):
            print("Loading existing XGBoost model...")
            self.model = joblib.load(self.model_path)
            
    def prepare_data(self, df):
        # Predict actual runtime based on requested resources and estimated runtime
        features = ["requested_cpu", "requested_gpu", "requested_memory", "estimated_runtime", "priority"]
        target = "actual_runtime"
        
        X = df[features]
        y = df[target]
        return X, y
        
    def train(self, data_path="data/synthetic_hpc_jobs.csv"):
        print(f"Training XGBoost predictor using {data_path}...")
        df = pd.read_csv(data_path)
        
        X, y = self.prepare_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        self.model.fit(X_train, y_train)
        
        predictions = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)
        
        print(f"Prediction Model Evaluation:\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR2: {r2:.2f}")
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")
        
    def predict(self, features):
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        return self.model.predict(features)
