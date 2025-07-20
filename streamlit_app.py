import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from src.data_preprocessing.cleaning import DataCleaner
from src.feature_engineering.feature_engineering import FeatureEngineer

# ✅ Load the trained model
MODEL_PATH = "outputs/models/best_model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error("❌ Trained model not found. Please run main_pipeline.py first.")
    st.stop()

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
expected_features = model_data["feature_names"]

st.success("✅ Model loaded successfully!")

# 🧹 Initialize preprocessors
cleaner = DataCleaner()
engineer = FeatureEngineer()

CATEGORY_MAP = {
    1: "Film & Animation", 2: "Autos & Vehicles", 10: "Music", 15: "Pets & Animals",
    17: "Sports", 18: "Short Movies", 19: "Travel & Events", 20: "Gaming",
    21: "Videoblogging", 22: "People & Blogs", 23: "Comedy", 24: "Entertainment",
    25: "News & Politics", 26: "Howto & Style", 27: "Education", 28: "Science & Technology",
    29: "Nonprofits & Activism", 30: "Movies", 31: "Anime/Animation", 32: "Action/Adventure",
    33: "Classics", 34: "Comedy (Film)", 35: "Documentary", 36: "Drama", 37: "Family",
    38: "Foreign", 39: "Horror", 40: "Sci-Fi/Fantasy", 41: "Thriller", 42: "Shorts",
    43: "Shows", 44: "Trailers"
}

st.title("🎥 YouTube Video View Count Predictor")
st.markdown("Upload a raw YouTube CSV, use default YouTube API data, or manually enter video features.")

# 🔧 Common preprocessing function
def preprocess(df):
    df = cleaner.clean_data(df)
    df = cleaner.encode_categorical_features(df, ['category_id'])
    if 'view_count' in df.columns:
        df = df.drop(columns=['view_count'])
    df = engineer.engineer_all(df)
    drop_cols = ['video_id', 'title', 'description', 'tags', 'publish_time', 'duration']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    return df

# 🔘 Manual Entry
option = st.radio("Choose input method:", ["Manual Entry"])

if option == "Manual Entry":
    st.subheader("🛠️ Enter Features Manually")

    title = st.text_input("📝 Title Text", value="Sample YouTube title")
    description = st.text_area("📄 Description Text", value="This is a sample YouTube video description.")
    tags = st.text_input("🏷️ Tags (comma separated)", value="tag1, tag2")
    publish_time = st.text_input("🕒 Publish Time (YYYY-MM-DD HH:MM:SS)", value="2024-01-01 12:00:00")
    duration = st.text_input("⏱️ Video Duration (e.g. PT5M30S)", value="PT5M30S")

    like_count = st.number_input("👍 Likes", min_value=0)
    comment_count = st.number_input("💬 Comments", min_value=0)
    category_name = st.selectbox("🎯 Category", list(CATEGORY_MAP.values()))
    category_id = [k for k, v in CATEGORY_MAP.items() if v == category_name][0]

    input_raw_df = pd.DataFrame([{
        "video_id": "manual_123",
        "title": title,
        "description": description,
        "tags": tags,
        "like_count": like_count,
        "comment_count": comment_count,
        "category_id": category_id,
        "publish_time": publish_time,
        "duration": duration,
        "view_count": 0
    }])

    if st.button("🔍 Predict"):
        try:
            processed = preprocess(input_raw_df)

            # 🧪 Debug info
            st.write("🧪 Model expects features:", expected_features)
            st.write("🧪 Processed columns:", list(processed.columns))

            processed = processed[expected_features]
            pred_raw = model.predict(processed)[0]
            st.write(f"🔎 Raw Prediction (log view count): {pred_raw:.4f}")

            pred = max(0, int(np.expm1(pred_raw)))
            st.success(f"📈 Predicted View Count: **{pred:,} views**")

        except Exception as e:
            st.error(f"❌ Error during manual prediction: {e}")
