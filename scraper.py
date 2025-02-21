import os
from dotenv import load_dotenv
from newsapi import NewsApiClient

load_dotenv()  # Load environment variables from .env
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def fetch_news(country="in", category="general"):
    """
    Fetch news from NewsAPI.org for the specified country and category.
    Returns a list of dictionaries with title, content, and URL.
    """
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    response = newsapi.get_top_headlines(language="en", country=country, category=category, page_size=5)
    
    if response.get("status") == "ok":
        articles = response.get("articles", [])
        return [
            {
                "title": article.get("title"),
                "content": article.get("description", "No content available"),
                "url": article.get("url")
            }
            for article in articles
        ]
    else:
        print("Error fetching news:", response)
        return []

if __name__ == "__main__":
    news = fetch_news("in", "general")
    for item in news:
        print(item)
