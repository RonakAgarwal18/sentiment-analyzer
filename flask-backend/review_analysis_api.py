from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from keybert import KeyBERT
from collections import Counter
from flask_cors import CORS
import nltk
import json

nltk.download('vader_lexicon')

# Initialize components
app = Flask(__name__)
CORS(app)
sia = SentimentIntensityAnalyzer()
kw_model = KeyBERT()

# Aspect Keywords
aspect_keywords = {
    "Delivery": ["delivered", "delivery", "late", "delay", "arrived", "on time", "fast", "shipping", "courier"],
    "Product Quality": ["quality", "material", "defective", "broken", "scratch", "durable", "cheap", "faulty", "poor", "damaged", "not working", "excellent", "perfect", "bad", "worth", "useless", "loves tablet", "love echo", "tablet good", "battery life", "best tablet", "love alexa", "nice tablet", "great product", "tablet perfect", "perfect tablet", "good product", "excellent tablet", "great kindle", "love tablet", "kindle great", "love kindle", "good tablet", "great tablet", "tablet great"],
    "Customer Support": ["customer service", "support", "help", "representative", "call", "email", "contact", "response"],
    "Pricing": ["expensive", "cheap", "value", "money", "cost", "pricing", "worth", "deal", "pricey"],
    "Packaging": ["packaging", "box", "sealed", "unboxing", "damaged box", "wrap", "tape", "cover"]
}

# Helper functions
def get_sentiment(review):
    score = sia.polarity_scores(review)['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def assign_aspect(keywords_list):
    if not isinstance(keywords_list, list) or not keywords_list:
        return "Other"
    keywords_str = " ".join(keywords_list).lower()
    for aspect, kw_list in aspect_keywords.items():
        for kw in kw_list:
            if kw in keywords_str:
                return aspect
    return "Other"

# API route
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    reviews = data.get('reviews', [])

    if not reviews:
        return jsonify({"error": "No reviews provided"}), 400

    # Create DataFrame
    reviews_df = pd.DataFrame({'review': reviews})

    # Sentiment
    reviews_df['sentiment'] = reviews_df['review'].apply(get_sentiment)

    # Keywords
    reviews_df['keywords'] = reviews_df['review'].apply(
        lambda x: [kw[0] for kw in kw_model.extract_keywords(x, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=2)]
    )

    # Aspect classification
    reviews_df['aspect'] = reviews_df['keywords'].apply(assign_aspect)

    # Summary
    insights = {}
    for aspect in reviews_df['aspect'].unique():
        subset = reviews_df[reviews_df['aspect'] == aspect]
        sentiments = subset['sentiment'].value_counts(normalize=True).to_dict()
        sentiments = {k: round(v * 100, 2) for k, v in sentiments.items()}

        all_keywords = " ".join(subset['keywords'].dropna().astype(str)).lower().split()
        common_keywords = [kw for kw, _ in Counter(all_keywords).most_common(5)]

        negative = sentiments.get('Negative', 0)
        positive = sentiments.get('Positive', 0)

        if negative >= 30:
            insight = f"\u26a0\ufe0f Many customers are unhappy with {aspect.lower()} (negative sentiment: {negative}%)."
        elif positive >= 70:
            insight = f"\u2705 Customers are highly satisfied with {aspect.lower()} (positive sentiment: {positive}%)."
        else:
            insight = f"\ud83d\udccc Mixed feedback on {aspect.lower()} (positive: {positive}%, negative: {negative}%)."

        insights[aspect] = {
            "sentiment": sentiments,
            "common_keywords": common_keywords,
            "insight": insight
        }

    response = {
        "reviews_analysis": reviews_df.to_dict(orient='records'),
        "insights": insights
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
