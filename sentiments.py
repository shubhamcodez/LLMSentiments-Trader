from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

# Load the RoBERTa model fine-tuned for financial sentiment
model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def analyze_financial_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)
    
    sentiment_mapping = {0: "negative", 1: "neutral", 2: "positive"}
    predicted_class = torch.argmax(logits, dim=1).item()
    sentiment = sentiment_mapping[predicted_class]
    
    scores = {label: prob.item() for label, prob in zip(sentiment_mapping.values(), probabilities[0])}
    
    return sentiment, scores

# Example usage
text = "Inflation is 10 percent up, consumers crying"
sentiment, scores = analyze_financial_sentiment(text)
print(f"Sentiment: {sentiment}")
print(f"Scores: Negative: {scores['negative']:.4f}, Neutral: {scores['neutral']:.4f}, Positive: {scores['positive']:.4f}")