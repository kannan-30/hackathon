from flask import Flask, request, jsonify
import pickle
from utils.preprocess import preprocess_text

app = Flask(__name__)

# Load Best Model & Vectorizer
with open("models/best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['text']
    processed_text = preprocess_text(data)
    text_vector = vectorizer.transform([processed_text])
    prediction = best_model.predict(text_vector)[0]
    result = "Fake News" if prediction == 1 else "Real News"
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
