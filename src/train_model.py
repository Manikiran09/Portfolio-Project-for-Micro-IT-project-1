import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from preprocess import preprocess_dataframe

def load_data(filepath):
    """Load the dataset CSV."""
    df = pd.read_csv(filepath)
    df = df[df['airline_sentiment'].isin(['positive', 'negative', 'neutral'])]  # Keep valid labels
    df = preprocess_dataframe(df, text_column='text')  # Preprocess tweets
    return df[['clean_text', 'airline_sentiment']]

def train_model(X_train, y_train):
    """Train Logistic Regression model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def main():
    # Step 1: Load and preprocess data
    df = load_data('data/Tweets.csv')

    # Step 2: Split dataset
    X = df['clean_text']
    y = df['airline_sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Step 4: Train model
    model = train_model(X_train_tfidf, y_train)

    # Step 5: Evaluate
    y_pred = model.predict(X_test_tfidf)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Step 6: Save model and vectorizer
    joblib.dump(model, 'models/sentiment_model.pkl')
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
    print("Model and vectorizer saved in 'models/' directory.")

if __name__ == "__main__":
    main()
