# Portfolio-Project-for-Micro-IT-project-1
# Twitter Sentiment Analysis

This project builds a machine learning pipeline to classify the sentiment of tweets as **positive**, **neutral**, or **negative**. It includes data preprocessing, model training, and prediction scripts.

---

# Project Structure
sentiment-analysis/
│
├── data/ # Raw and processed datasets
│ └── Tweets.csv # Original dataset CSV
│
├── notebooks/ # Jupyter notebooks for EDA and experimentation
│ └── EDA.ipynb # Exploratory Data Analysis and insights
│
├── models/ # Saved models and vectorizers
│ ├── sentiment_model.pkl # Trained sentiment classification model
│ └── tfidf_vectorizer.pkl # TF-IDF vectorizer used for feature extraction
│
├── src/ # Source code modules
│ ├── preprocess.py # Text cleaning and preprocessing functions
│ ├── train_model.py # Model training script
│ └── predict.py # Load model and vectorizer for predictions
│
├── main.py # Command-line interface for sentiment prediction
├── requirements.txt # Python dependencies
└── README.md # Project documentation (this file)

---


---
```bash
#Setup Instructions

1.Clone the repository**
git clone <repository_url>
cd sentiment-analysis
2.Installing and set up virtual environment
 python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
3.Installing Dependencies
pip install -r requirements.txt
4.Prepare your data

Place your dataset file (Tweets.csv) inside the data/ folder. The dataset should have at least the columns: text (tweet content) and airline_sentiment (labels: positive, neutral, negative).

Usage
Exploratory Data Analysis (EDA)
The notebooks/EDA.ipynb notebook contains exploratory data analysis, visualizations, and insights about the dataset to better understand sentiment distribution, common words, and data quality before training the model.

You can open and run this notebook in Jupyter:

jupyter notebook notebooks/EDA.ipynb

Train the model
Run the training script to preprocess data, train a Logistic Regression model, evaluate it, and save the model and vectorizer:
python src/train_model.py
Predict sentiment (CLI)
Run the main script 
python main.py
