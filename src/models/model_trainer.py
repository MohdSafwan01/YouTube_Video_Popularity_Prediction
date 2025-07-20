import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class ModelTrainer:
    def __init__(self):
        self.models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(random_state=42),
            "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=42)
        }
        self.best_model = None
        self.feature_names = None

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        results = []

        # ✅ Apply log1p to target to handle skewness
        y_train_log = np.log1p(y_train)
        y_test_log = np.log1p(y_test)

        # ✅ Save feature names for prediction use
        self.feature_names = list(X_train.columns)

        for name, model in self.models.items():
            model.fit(X_train, y_train_log)
            preds_log = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test_log, preds_log))
            r2 = r2_score(y_test_log, preds_log)
            mae = mean_absolute_error(y_test_log, preds_log)

            results.append({
                "Model": name,
                "RMSE": rmse,
                "R²": r2,
                "MAE": mae
            })
            print(f"✅ {name}: RMSE={rmse:.2f}, R²={r2:.2f}, MAE={mae:.2f}")

        df_results = pd.DataFrame(results).sort_values("R²", ascending=False)
        best_model_name = df_results.iloc[0]["Model"]
        self.best_model = self.models[best_model_name]
        print(f"\n🎯 Best model: {best_model_name}")
        return df_results

    def save_model(self, output_path="outputs/models/best_model.pkl"):
        if self.best_model:
            joblib.dump({
                "model": self.best_model,
                "feature_names": self.feature_names
            }, output_path)
            print(f"📦 Model and feature names saved to: {output_path}")
        else:
            print("❌ No model to save. Train a model first.")
