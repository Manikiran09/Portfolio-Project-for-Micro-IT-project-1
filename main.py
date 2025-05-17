# main.py

from src.predict import predict_sentiment

def main():
    # Example tweets for testing
    examples = [
        "I love the in-flight service and friendly staff!",
        "Worst airline experience ever. Never flying with them again!",
        "The flight was okay, nothing special.",
    ]

    for text in examples:
        sentiment = predict_sentiment(text)
        print(f"Tweet: {text}")
        print(f"Predicted Sentiment: {sentiment}\n")

if __name__ == "__main__":
    main()
