from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

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
    
    # Get the predicted label
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # Map the predicted class index to the sentiment label
    sentiment_mapping = {0: "negative", 1: "neutral", 2: "positive"}
    sentiment = sentiment_mapping[predicted_class]
    
    return sentiment

# Example usage
text = "The stock price increased significantly after the announcement of the merger."
sentiment = analyze_sentiment(text)
print(f"Sentiment: {sentiment}")
