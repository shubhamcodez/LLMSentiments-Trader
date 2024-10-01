# Real-Time NER and Sentiment Analysis for Financial News

**Authors**: Tianqi Wang, Shubham Singh

## Summary

FinSentify is a real-time news processing application designed for Named Entity Recognition (NER) and sentiment analysis, leveraging large language models (LLMs) like DistilBERT and FinBERT. This project aims to provide actionable insights for hedge funds by analyzing news data, enabling rapid investment decisions.

## Relevance

In mid- to high-frequency trading, speed and accuracy are paramount. Traditional NER and sentiment analysis methods like regular expression matching and Conditional Random Fields (CRFs) provide reasonable results but can fall short compared to modern LLMs. DistilBERT, a lighter variant of BERT, offers an ideal solution by retaining much of BERT's performance while enhancing processing speed. This project focuses on harnessing DistilBERT for timely investment decisions based on financial news sentiment.

## Methodology

### Data

- **Dataset**: Opoint news feed (20GB of global news articles)
- **Timeframe**: Collected between May 12, 2024, 10:00 PM and May 13, 2024, 10:00 PM
- **Focus**: U.S. market articles in English
- **Labeling**: The dataset includes partially labeled entities and sentiment, serving as a benchmark for analysis.

### Portfolio Construction

- **Strategy**: Capture stock price movements triggered by news events
- **Decision-making**: Trade based on sentiment analysis (long for positive sentiment, short for negative, hold for neutral)
- **Holding period**: Each trade is held for one hour

### Insights

The analysis reveals a high volume of articles published, particularly during peak hours (1 PM - 2 AM UTC). We filtered for articles from the top 40 news agencies, focusing on those likely to impact the market.

## Findings

- **Starting Balance**: $200,000
- **Ending Balance**: $200,419 (0.21% return)
- **Performance Metrics**:
    - Sharpe Ratio: 0.84 (benchmark: 0.16)
    - Max Drawdown: -0.22% (benchmark: -0.30%)

The strategy demonstrates that trading based on sentiment analysis can outperform traditional methods, especially when leveraging advanced NLP techniques.

## Conclusion

FinSentify showcases the potential of combining NER and sentiment analysis using LLMs within trading strategies. While initial results are promising, future work should focus on expanding the dataset, fine-tuning models, and implementing risk management techniques to enhance returns.

## Future Improvements

1. Expand to include small-cap and mid-cap stocks.
2. Utilize a broader dataset and integrate additional data sources.
3. Implement advanced risk management techniques.

## Installation

```bash
git clone https://github.com/yourusername/FinSentify.git
cd FinSentify
pip install -r requirements.txt
