import requests
from urllib.parse import urlparse, parse_qs

class YouTubeAPILoader:
    def __init__(self, api_key):
        self.api_key = api_key

    def extract_video_id(self, url):
        query = urlparse(url)
        if query.hostname == 'youtu.be':
            return query.path[1:]
        elif query.hostname in ('www.youtube.com', 'youtube.com'):
            if query.path == '/watch':
                return parse_qs(query.query)['v'][0]
            elif query.path.startswith('/embed/'):
                return query.path.split('/')[2]
            elif query.path.startswith('/v/'):
                return query.path.split('/')[2]
        return None

    def get_video_details(self, video_url):
        video_id = self.extract_video_id(video_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")

        url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics,contentDetails&id={video_id}&key={self.api_key}"
        response = requests.get(url)
        data = response.json()

        if not data.get("items"):
            raise Exception("Video not found or API quota exceeded.")

        item = data["items"][0]
        snippet = item["snippet"]
        stats = item["statistics"]
        content = item["contentDetails"]

        return {
            "video_id": video_id,
            "title": snippet.get("title", ""),
            "description": snippet.get("description", ""),
            "tags": ", ".join(snippet.get("tags", [])),
            "publish_time": snippet.get("publishedAt", "2024-01-01T00:00:00Z"),
            "duration": content.get("duration", "PT0M0S"),
            "category_id": snippet.get("categoryId", 0),
            "like_count": int(stats.get("likeCount", 0)),
            "comment_count": int(stats.get("commentCount", 0)),
            "view_count": int(stats.get("viewCount", 0)),
            "subscriber_count": 0  # YouTube doesn't return this in video data
        }
