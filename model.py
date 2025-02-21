import os
import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import resample

def simple_preprocess(text):
    """
    Simple preprocessing:
    - Lowercase the text
    - Remove extra whitespace
    (Punctuation is kept to preserve potential signals)
    """
    return " ".join(text.lower().split())

def train_and_save_model():
    # Load datasets (ensure Fake.csv and True.csv are in the same folder)
    try:
        fake_df = pd.read_csv("fake_news.csv")
        true_df = pd.read_csv("true_news.csv")
    except Exception as e:
        print("Error loading CSV files:", e)
        return

    # Set labels: Fake = 1, Real = 0
    fake_df["label"] = 1
    true_df["label"] = 0

    # Print initial counts
    print("Initial counts:")
    print("Fake:", len(fake_df))
    print("Real:", len(true_df))

    # Balance the dataset using resampling (downsample the larger class)
    min_count = min(len(fake_df), len(true_df))
    fake_df_bal = resample(fake_df, replace=False, n_samples=min_count, random_state=42)
    true_df_bal = resample(true_df, replace=False, n_samples=min_count, random_state=42)

    # Combine and shuffle the datasets
    df = pd.concat([fake_df_bal, true_df_bal], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("\nAfter balancing, label distribution:")
    print(df['label'].value_counts())

    # Replace any "[empty]" placeholders with an empty string
    df["title"] = df["title"].replace("[empty]", "")
    df["text"] = df["text"].replace("[empty]", "")

    # Combine title and text into a single column
    df["combined_text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()

    # Drop rows where combined_text is empty
    before_drop = len(df)
    df = df[df["combined_text"] != ""]
    print(f"\nDropped {before_drop - len(df)} rows with empty combined text.")

    # Apply simple preprocessing
    df["combined_text"] = df["combined_text"].apply(simple_preprocess)

    # Print a few preprocessed samples for inspection
    print("\nSample preprocessed texts:")
    print(df["combined_text"].head(5))

    # Prepare features and labels
    X = df["combined_text"]
    y = df["label"]

    # Split the data (with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build a pipeline: TF-IDF vectorizer + Logistic Regression with balanced class weights
    pipeline = make_pipeline(
        TfidfVectorizer(stop_words="english"),
        LogisticRegression(max_iter=300, class_weight='balanced')
    )

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate the model on the test set
    accuracy = pipeline.score(X_test, y_test)
    print(f"\nModel Accuracy on test set: {accuracy:.2f}")
    y_pred = pipeline.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Ensure the "models" directory exists
    os.makedirs("models", exist_ok=True)

    # Save the trained pipeline
    joblib.dump(pipeline, "models/best_model.pkl")
    print("\nModel saved to models/best_model.pkl")

if __name__ == "__main__":
    train_and_save_model()
