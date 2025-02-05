# LLMSentiments Trader

## Real-Time NER and Sentiment Analysis for Financial News

**Authors**: Tianqi Wang, Shubham Singh

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![DistilBERT](https://img.shields.io/badge/DistilBERT-Transformer-yellow)
![FinBERT](https://img.shields.io/badge/FinBERT-Financial%20NLP-green)
![CSP](https://img.shields.io/badge/CSP-Stream%20Processing-red)

---

## Summary

**LLMSentiments Trader** is a cutting-edge application that leverages large language models (LLMs) like **DistilBERT** and **FinBERT** for real-time Named Entity Recognition (NER) and sentiment analysis of financial news. Designed to provide actionable insights for hedge funds, this project enables rapid investment decisions based on news data analysis. We've integrated the **CSP library** by Point72 for high-speed stream processing, enhancing the overall efficiency of our application.

---

## Features

- Real-time processing of financial news
- Named Entity Recognition (NER) using DistilBERT
- Sentiment analysis using FinBERT
- High-speed stream processing with the CSP library
- Portfolio construction based on sentiment analysis

---

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

It showcases the potential of combining NER and sentiment analysis using LLMs within trading strategies. While initial results are promising, future work should focus on expanding the dataset, fine-tuning models, and implementing risk management techniques to enhance returns.

## Future Improvements

1. Expand to include small-cap and mid-cap stocks.
2. Utilize a broader dataset and integrate additional data sources.
3. Implement advanced risk management techniques.

## Installation

```bash
git clone https://github.com/shubhamcodez/LLMSentiments-Trader.git
cd LLMSentiments-Trader
pip install -r requirements.txt
