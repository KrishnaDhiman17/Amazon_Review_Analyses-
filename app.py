import os
import re
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model and vectorizer safely
def load_assets():
    try:
        model = joblib.load("sentiment_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

model, vectorizer = load_assets()

def clean_text(text):
    if not text: return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    review = ""

    if request.method == "POST":
        review = request.form.get("review", "")
        if review and model and vectorizer:
            cleaned = clean_text(review)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            sentiment = prediction.capitalize()

    return render_template("index.html", sentiment=sentiment, review=review)

if __name__ == "__main__":
    app.run(debug=True)