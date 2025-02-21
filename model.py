import os
import re
import string
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import nltk

# Ensure stopwords are downloaded
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# File paths (CSV files are expected in the same directory as model.py)
FAKE_NEWS_PATH = os.path.join(os.path.dirname(__file__), "fake_news.csv")
TRUE_NEWS_PATH = os.path.join(os.path.dirname(__file__), "true_news.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")

def clean_text(text: str) -> str:
    """
    Lowercase, remove punctuation, extra whitespace, and stopwords.
    """
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words("english"))
    return " ".join(word for word in text.split() if word not in stop_words)

def load_and_preprocess_data():
    """
    Load fake and true news CSVs, clean the text, and assign labels.
    """
    fake_df = pd.read_csv(FAKE_NEWS_PATH)
    true_df = pd.read_csv(TRUE_NEWS_PATH)
    
    # Assume the CSV files have a 'text' column for article content
    fake_df["clean_text"] = fake_df["text"].apply(clean_text)
    true_df["clean_text"] = true_df["text"].apply(clean_text)
    
    # Label data: Fake = 0, Real = 1
    fake_df["label"] = 0
    true_df["label"] = 1
    
    # Combine and shuffle
    df = pd.concat([fake_df, true_df]).reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def train_and_select_model():
    """
    Train RandomForest, GradientBoosting, and XGBoost classifiers.
    Evaluate them, select the best based on accuracy, and save the model and vectorizer.
    """
    df = load_and_preprocess_data()
    X = df["clean_text"]
    y = df["label"]
    
    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)
    
    # Split data for training and evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
    
    # Define the models to evaluate
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    }
    
    best_model = None
    best_accuracy = 0
    best_model_name = ""
    
    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"{name} Accuracy: {acc:.4f}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name
            
    print(f"Selected Model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    
    # Save the best model and the vectorizer
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    return best_model, vectorizer

def load_model():
    """
    Load the best model and vectorizer from disk.
    If not found, train and select the best model.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("Model or vectorizer not found. Training a new model...")
        return train_and_select_model()
    else:
        best_model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("Best model and vectorizer loaded successfully.")
        return best_model, vectorizer

# Load (or train) the model when this module is imported
model, vectorizer = load_model()

def predict_news(news_text: str) -> str:
    """
    Clean and vectorize the input text, then predict whether it is Fake or Real.
    """
    cleaned_text = clean_text(news_text)
    news_vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(news_vectorized)[0]
    return "Real" if prediction == 1 else "Fake"

if __name__ == "__main__":
    sample_text = "Breaking news: Major breakthrough in renewable energy technology."
    print("Prediction:", predict_news(sample_text))
