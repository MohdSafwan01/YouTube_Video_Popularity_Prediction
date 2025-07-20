from googleapiclient.discovery import build
import pandas as pd
import time

class YouTubeAPILoader:
    def __init__(self, api_key="AIzaSyBsqf1tgzHNQXiRCF9YQ-pwyCJ7Jc4_RpU"):
        self.api_key = api_key
        self.youtube = build("youtube", "v3", developerKey=self.api_key)

    def search_video_ids(self, query, max_results=20):
        search_response = self.youtube.search().list(
            q=query,
            part="id",
            type="video",
            maxResults=max_results
        ).execute()

        video_ids = [item["id"]["videoId"] for item in search_response["items"]]
        return video_ids

    def get_video_details(self, video_ids):
        video_details = []

        for i in range(0, len(video_ids), 50):
            response = self.youtube.videos().list(
                part="snippet,statistics,contentDetails",
                id=",".join(video_ids[i:i+50])
            ).execute()

            for item in response["items"]:
                video = {
                    "video_id": item["id"],
                    "title": item["snippet"].get("title"),
                    "description": item["snippet"].get("description"),
                    "tags": "|".join(item["snippet"].get("tags", [])),
                    "category_id": item["snippet"].get("categoryId"),
                    "publish_time": item["snippet"].get("publishedAt"),
                    "view_count": item["statistics"].get("viewCount"),
                    "like_count": item["statistics"].get("likeCount"),
                    "comment_count": item["statistics"].get("commentCount"),
                    "duration": item["contentDetails"].get("duration")
                }
                video_details.append(video)
            time.sleep(1)  # Respect API rate limits

        return pd.DataFrame(video_details)

    def fetch_data(self, query="AI coding", max_results=20):
        ids = self.search_video_ids(query, max_results)
        df = self.get_video_details(ids)
        return df
