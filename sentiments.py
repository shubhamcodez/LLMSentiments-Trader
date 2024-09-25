from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load the FinBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# Function to analyze sentiment
def analyze_sentiment(text):
    # Tokenize the input text and get the input tensors
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Run the model on the input text
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted label and apply softmax to get probabilities
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)
    
    # Extract probabilities for each sentiment
    negative_score = probabilities[0][0].item()
    neutral_score = probabilities[0][1].item()
    positive_score = probabilities[0][2].item()
    
    # Map the predicted class index to the sentiment label
    sentiment_mapping = {0: "negative", 1: "neutral", 2: "positive"}
    predicted_class = torch.argmax(logits, dim=1).item()
    sentiment = sentiment_mapping[predicted_class]
    
    return sentiment, negative_score, neutral_score, positive_score

# Example usage
text = "The stock price increased significantly after the announcement of the merger."
sentiment, neg, neu, pos = analyze_sentiment(text)
print(f"Sentiment: {sentiment}")
print(f"Scores -> Negative: {neg}, Neutral: {neu}, Positive: {pos}")
