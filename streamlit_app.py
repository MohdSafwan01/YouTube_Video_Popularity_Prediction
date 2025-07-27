import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from src.data_preprocessing.cleaning import DataCleaner
from src.feature_engineering.feature_engineering import FeatureEngineer
from fetch_youtube_data import YouTubeAPILoader

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
    st.write("🟡 Initial shape:", df.shape)

    df = cleaner.clean_data(df)
    st.write("🔵 After cleaning:", df.shape)

    df = cleaner.encode_categorical_features(df, ['category_id'])
    st.write("🟣 After encoding category_id:", df.shape)

    if 'view_count' in df.columns:
        df = df.drop(columns=['view_count'])

    df = engineer.engineer_all(df)
    st.write("🟠 After feature engineering:", df.shape)

    drop_cols = ['video_id', 'title', 'description', 'tags', 'publish_time', 'duration']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    st.write("✅ Final shape before prediction:", df.shape)
    return df

# 🔘 Input Method Selector
option = st.radio("Choose input method:", ["Manual Entry", "Upload CSV File", "YouTube URL"])

# ✍️ Manual Entry
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
            processed = processed[expected_features]
            pred_raw = model.predict(processed)[0]
            pred = max(0, int(np.expm1(pred_raw)))
            st.success(f"📈 Predicted View Count: **{pred:,} views**")
        except Exception as e:
            st.error(f"❌ Error during manual prediction: {e}")

# 📂 Upload CSV File
elif option == "Upload CSV File":
    st.subheader("📂 Upload YouTube Data CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("📄 Uploaded Data Preview:")
            st.dataframe(df.head())

            # Save original index for final merge if needed
            original_df = df.copy()

            # 🧹 Preprocess
            processed = preprocess(df)

            # 🔍 Predict only for rows that survived preprocessing
            pred_raw = model.predict(processed)
            pred_views = [max(0, int(np.expm1(p))) for p in pred_raw]

            # 📊 Add predictions back to cleaned data
            processed["predicted_views"] = pred_views

            st.write("✅ Predictions:")
            st.dataframe(processed)

        except Exception as e:
            st.error(f"❌ Error processing CSV file: {str(e)}")

# 🌐 YouTube URL + API
elif option == "YouTube URL":
    st.subheader("🎥 Predict from YouTube Video URL")
    api_key = st.text_input("🔑 Enter YouTube Data API Key", type="password")
    video_url = st.text_input("📺 Enter YouTube Video URL")

    if st.button("Fetch & Predict"):
        if api_key and video_url:
            try:
                yt_loader = YouTubeAPILoader(api_key)
                video_data = yt_loader.get_video_details(video_url)

                st.write("📋 Fetched video metadata:")
                st.json(video_data)

                input_raw_df = pd.DataFrame([{
                    "video_id": "api_123",
                    "title": video_data.get("title", ""),
                    "description": "",
                    "tags": "",
                    "like_count": int(video_data.get("like_count", 0)),
                    "comment_count": int(video_data.get("comment_count", 0)),
                    "category_id": int(video_data.get("category_id", 0)),
                    "publish_time": "2024-01-01 12:00:00",
                    "duration": "PT5M30S",
                    "view_count": 0
                }])

                processed = preprocess(input_raw_df)
                processed = processed[expected_features]
                pred_raw = model.predict(processed)[0]
                pred = max(0, int(np.expm1(pred_raw)))

                st.success(f"📈 Predicted View Count: **{pred:,} views**")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please provide both API key and video URL.")
