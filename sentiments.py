import json
import codecs
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm

MODEL_NAME = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
LOCAL_MODEL_PATH = "./financial_sentiment_model"
LOCAL_TOKENIZER_PATH = "./financial_sentiment_tokenizer"
DATASET_PATH = "english_data.jsonl"
OUTPUT_PATH = "opoint_sentiment_analysis_results_1000.json"
MAX_ARTICLES = 1000

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

def analyze_financial_sentiment(model, tokenizer, text):
    try:
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
    except Exception as e:
        print(f"Error in analyze_financial_sentiment: {str(e)}")
        return "error", {"error": str(e)}

def process_opoint_dataset(model, tokenizer):
    results = []
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with codecs.open(DATASET_PATH, 'r', encoding=encoding) as file:
                for line in tqdm(file, total=MAX_ARTICLES, desc="Processing articles"):
                    try:
                        article = json.loads(line)
                        
                        # Extract relevant fields according to Opoint structure
                        body = article.get('body', {}).get('text', '')
                        header = article.get('header', {}).get('text', '')
                        summary = article.get('summary', {}).get('text', '')
                        
                        # Combine header, summary, and body for sentiment analysis
                        full_text = f"{header}\n\n{summary}\n\n{body}"
                        
                        if full_text:
                            sentiment, scores = analyze_financial_sentiment(model, tokenizer, full_text)
                            
                            result = {
                                'id_article': article.get('id_article'),
                                'id_site': article.get('id_site'),
                                'sentiment': sentiment,
                                'scores': scores,
                                'header': header, #[:100] + '...' if len(header) > 100 else header,
                                'summary': summary, #[:100] + '...' if len(summary) > 100 else summary,
                                'body_snippet': body, #[:100] + '...' if len(body) > 100 else body,
                                'unix_timestamp': article.get('unix_timestamp'),
                                'language': article.get('language', {}).get('text'),
                                'countrycode': article.get('countrycode'),
                                'url': article.get('url'),
                                'orig_url': article.get('orig_url'),
                            }
                            results.append(result)
                        else:
                            print(f"Skipping article {article.get('id_article', 'N/A')}: Invalid or empty content")
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in line: {line[:100]}...")
                        continue
                    except Exception as e:
                        print(f"Error processing article: {str(e)}")
                        continue
                    
                    if len(results) >= MAX_ARTICLES:
                        break
                
                print(f"\nSuccessfully processed {len(results)} articles using {encoding} encoding.")
                break  # If successful, exit the loop
        except UnicodeDecodeError:
            print(f"Failed to read with {encoding} encoding. Trying next...")
    else:
        print("Failed to read the file with all attempted encodings.")
    
    return results

def save_results(results):
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    model, tokenizer = load_or_download_model()
    results = process_opoint_dataset(model, tokenizer)
    save_results(results)
    print(f"Analysis complete. Results saved to '{OUTPUT_PATH}'")
    
    # Print a summary of the results
    sentiments = [r['sentiment'] for r in results]
    print("\nSentiment Distribution:")
    print(f"Positive: {sentiments.count('positive')}")
    print(f"Neutral: {sentiments.count('neutral')}")
    print(f"Negative: {sentiments.count('negative')}")
    print(f"Errors: {sentiments.count('error')}")

if __name__ == "__main__":
    main()