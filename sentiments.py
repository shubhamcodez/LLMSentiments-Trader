from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import os

MODEL_NAME = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
LOCAL_MODEL_PATH = "./financial_sentiment_model"
LOCAL_TOKENIZER_PATH = "./financial_sentiment_tokenizer"

def load_or_download_model():
    if os.path.exists(LOCAL_MODEL_PATH) and os.path.exists(LOCAL_TOKENIZER_PATH):
        print("Loading model and tokenizer from local storage...")
        model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_TOKENIZER_PATH)
    else:
        print("Downloading model and tokenizer...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        print("Saving model and tokenizer locally...")
        model.save_pretrained(LOCAL_MODEL_PATH)
        tokenizer.save_pretrained(LOCAL_TOKENIZER_PATH)
    
    return model, tokenizer

model, tokenizer = load_or_download_model()

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