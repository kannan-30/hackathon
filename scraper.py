import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NEWSAPI_KEY")
BASE_URL = "https://newsapi.org/v2/top-headlines"

def fetch_news(country="in", category="general"):
    """
    Fetch news from NewsAPI.org for the specified country and category.
    """
    params = {"country": country, "category": category, "apiKey": API_KEY}
    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        articles = response.json().get("articles", [])
        return [
            {
                "title": article.get("title"),
                "content": article.get("description", "No content available"),
                "url": article.get("url")
            }
            for article in articles
        ]
    return []

if __name__ == "__main__":
    # Example: Fetch Indian news
    news = fetch_news("in")
    print(news)
