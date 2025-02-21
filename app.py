from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import string

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def preprocess_text(text):
    """
    Preprocess text (must match training steps):
    - Lowercase
    - Remove punctuation
    - Remove extra whitespace
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load the trained model
try:
    model = joblib.load("models/best_model.pkl")
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)

@app.route('/')
def home():
    return "Flask server is running! Use POST /predict to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get("text", "")
        if not data:
            return jsonify({"error": "No text provided"}), 400

        processed_text = preprocess_text(data)
        probabilities = model.predict_proba([processed_text])[0]
        prediction = model.predict([processed_text])[0]

        # Debug info in server logs:
        print("Input:", processed_text)
        print("Probabilities:", probabilities)
        print("Prediction:", prediction)

        # Label: Fake News = 1, Real News = 0
        result = "Fake News" if prediction == 1 else "Real News"
        return jsonify({"prediction": result, "probabilities": probabilities.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Available Routes:")
    print(app.url_map)
    app.run(debug=True)
