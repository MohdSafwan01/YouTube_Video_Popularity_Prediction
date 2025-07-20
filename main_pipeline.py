# ✅ 1. main_pipeline.py
from src.data_preprocessing.cleaning import DataCleaner
from src.feature_engineering.feature_engineering import FeatureEngineer
from src.models.model_trainer import ModelTrainer
from fetch_youtube_data import YouTubeAPILoader

import pandas as pd

# 🚀 Load Data from YouTube API
yt_loader = YouTubeAPILoader()
df = yt_loader.fetch_data(query="coding tutorial", max_results=50)

# 🧹 Clean Data
print("\n🧹 Cleaning data...")
cleaner = DataCleaner()
df_clean = cleaner.clean_data(df)
df_clean = cleaner.encode_categorical_features(df_clean, ['category_id'])

# 🧠 Feature Engineering
print("\n🛠️ Feature engineering...")
engineer = FeatureEngineer()
df_features = engineer.engineer_all(df_clean)
df_features = df_features[df_features['view_count'].notna()]
df_features['view_count'] = df_features['view_count'].astype(float)

# 🚫 Drop non-numeric features
cols_to_drop = ['video_id', 'title', 'description', 'tags', 'publish_time', 'duration']
df_features = df_features.drop(columns=[c for c in cols_to_drop if c in df_features.columns], errors='ignore')

# 📊 Split Data
X_train, X_test, y_train, y_test = cleaner.split_data(df_features, 'view_count')

# 🤖 Train Model
trainer = ModelTrainer()
results = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
trainer.save_model("outputs/models/best_model.pkl")

# ✅ Done
print("\n📊 Final Model Comparison:")
print(results)
