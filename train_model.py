import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

def train():
    # Load dataset
    try:
        data = pd.read_csv("amazon_reviews.csv")
    except FileNotFoundError:
        print("Error: amazon_reviews.csv not found.")
        return

    # Keep required columns
    data = data[['reviewText', 'overall']].dropna()

    # Convert rating to sentiment
    def rating_to_sentiment(rating):
        if rating >= 4: return "positive"
        if rating <= 2: return "negative"
        return None

    data["sentiment"] = data["overall"].apply(rating_to_sentiment)
    data = data.dropna(subset=["sentiment"])

    # Balance dataset (Adjust 'n' based on your CSV size)
    pos_count = len(data[data.sentiment == "positive"])
    neg_count = len(data[data.sentiment == "negative"])
    sample_size = min(pos_count, neg_count, 2000)

    positive = data[data.sentiment == "positive"].sample(n=sample_size, random_state=42)
    negative = data[data.sentiment == "negative"].sample(n=sample_size, random_state=42)
    data = pd.concat([positive, negative])

    # Preprocess
    data["reviewText"] = data["reviewText"].apply(clean_text)

    # Vectorization
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=8000)
    X = vectorizer.fit_transform(data["reviewText"])
    y = data["sentiment"]

    # Split & Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluation
    print(classification_report(y_test, model.predict(X_test)))

    # Save using joblib for better compatibility
    joblib.dump(model, "sentiment_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("Model and Vectorizer saved successfully!")

if __name__ == "__main__":
    train()