# src/preprocess.py

import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import nltk

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tokenizer = TweetTokenizer()

def clean_text(text):
    """Clean raw tweet text."""
    text = text.lower()  # Lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r"@\w+", '', text)  # Remove mentions
    text = re.sub(r"#\w+", '', text)  # Remove hashtags
    text = re.sub(r"[0-9]+", '', text)  # Remove numbers
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)  # Remove punctuation
    text = re.sub(r"\s+", ' ', text).strip()  # Remove extra whitespace
    return text

def tokenize_lemmatize(text):
    """Tokenize and lemmatize tweet text."""
    tokens = tokenizer.tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def preprocess_text(text):
    """Full preprocessing pipeline."""
    cleaned = clean_text(text)
    return tokenize_lemmatize(cleaned)

def preprocess_dataframe(df, text_column='text'):
    """Preprocess a DataFrame of tweets."""
    df = df.copy()
    df['clean_text'] = df[text_column].apply(preprocess_text)
    return df
