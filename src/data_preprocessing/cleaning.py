import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        try:
            df = pd.read_csv(self.filepath)
            print("✅ Data loaded successfully.")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

class DataCleaner:
 import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataCleaner:
    def clean_data(self, df):
        df = df.drop_duplicates()

        # ✅ Convert numeric columns to proper dtype
        numeric_cols = ['view_count', 'like_count', 'comment_count']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna()  # Drop any rows with NaNs after conversion
        return df

    def encode_categorical_features(self, df, cols):
        for col in cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        return df

    def split_data(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return train_test_split(X, y, test_size=0.2, random_state=42)
