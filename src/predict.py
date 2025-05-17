# src/predict.py

import joblib
from .preprocess import preprocess_text

# Load the trained model and TF-IDF vectorizer
model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

def predict_sentiment(text: str) -> str:
    """
    Predict the sentiment of a given tweet.
    
    Args:
        text (str): Raw tweet text.
    
    Returns:
        str: Predicted sentiment label (e.g., 'positive', 'neutral', 'negative').
    """
    cleaned_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    return prediction
