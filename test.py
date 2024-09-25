import csp
from csp import ts
import requests
import time
from datetime import datetime, timedelta

# NewsAPI configuration
API_KEY = '1d7d21bd-de08-48ca-a0b5-8978e09786a0'
BASE_URL = 'https://newsapi.org/v2/everything'

class NewsData(csp.Struct):
    title: str
    description: str
    publishedAt: str
    source: str

@csp.node
def news_fetcher() -> ts[NewsData]:
    with csp.state():
        last_fetch_time = ts.time(0)
    
    while True:
        current_time = ts.time.now()
        if (current_time - last_fetch_time).total_seconds() >= 60:  # Fetch every 60 seconds
            try:
                params = {
                    'apiKey': API_KEY,
                    'q': 'technology',  # Example query
                    'language': 'en',
                    'sortBy': 'publishedAt'
                }
                response = requests.get(BASE_URL, params=params)
                response.raise_for_status()
                articles = response.json()['articles']
                for article in articles:
                    yield csp.output(NewsData(
                        title=article['title'],
                        description=article.get('description', ''),
                        publishedAt=article['publishedAt'],
                        source=article['source']['name']
                    ))
                last_fetch_time = current_time
                print(f"Fetched {len(articles)} articles")
            except requests.RequestException as e:
                print(f"Error fetching news: {e}")
        yield ts.sleep(1)

@csp.node
def news_processor(article: ts[NewsData]) -> ts[str]:
    return f"New article: {article.title} (Source: {article.source})"

@csp.graph
def news_graph():
    news = news_fetcher()
    processed = news_processor(news)
    csp.print("Processed:", processed)

if __name__ == "__main__":
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=5)  # Run for 5 minutes
    csp.run(news_graph, starttime=start_time, endtime=end_time, realtime=True)
