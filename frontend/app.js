async function analyzeText() {
    const newsText = document.getElementById("newsInput").value.trim();
    const resultDiv = document.getElementById("result");
  
    if (!newsText) {
      resultDiv.textContent = "Please enter some text.";
      return;
    }
  
    try {
      // Call your Flask backend (adjust the URL/port if needed)
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: newsText })
      });
  
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
  
      const data = await response.json();
  
      if (data.error) {
        resultDiv.textContent = "Error: " + data.error;
      } else if (data.prediction) {
        resultDiv.textContent = "Prediction: " + data.prediction;
      } else {
        resultDiv.textContent = "No valid prediction returned.";
      }
    } catch (error) {
      console.error(error);
      resultDiv.textContent = "An error occurred. Please try again.";
    }
  }
  