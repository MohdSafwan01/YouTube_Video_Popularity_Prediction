import pandas as pd
from textblob import TextBlob

class FeatureEngineer:
    def __init__(self):
        pass

    def add_text_features(self, df):
        df['title_length'] = df['title'].apply(lambda x: len(str(x)))
        df['description_length'] = df['description'].apply(lambda x: len(str(x)))
        df['tag_count'] = df['tags'].apply(lambda x: len(str(x).split('|')))
        return df

    def add_time_features(self, df):
        df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
        df['publish_hour'] = df['publish_time'].dt.hour
        df['publish_dayofweek'] = df['publish_time'].dt.dayofweek
        df['is_weekend'] = df['publish_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        df['is_prime_time'] = df['publish_hour'].apply(lambda x: 1 if 18 <= x <= 22 else 0)
        return df

    def add_sentiment_features(self, df):
        df['title_sentiment'] = df['title'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        df['desc_sentiment'] = df['description'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        return df

    def engineer_all(self, df):
        df = self.add_text_features(df)
        df = self.add_time_features(df)
        df = self.add_sentiment_features(df)
        return df
