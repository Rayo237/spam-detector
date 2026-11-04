from flask import Flask, request, jsonify, render_template_string
import joblib

app = Flask(__name__)
model = joblib.load("spam_model.joblib")

PAGE = """
<!doctype html>
<title>Spam Detector</title>
<h2>Spam Detector</h2>
<form method="post" action="/">
  <textarea name="text" rows="6" cols="60" placeholder="Paste a message..." required>{{ text or '' }}</textarea><br>
  <button type="submit">Check</button>
</form>
{% if result %}<p><b>{{ result }}</b> ({{ prob }}% spam probability)</p>{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = prob = text = None
    if request.method == "POST":
        text = request.form["text"]
        p = model.predict_proba([text])[0][1]
        prob = round(p * 100, 1)
        result = "SPAM" if p >= 0.5 else "Not spam"
    return render_template_string(PAGE, result=result, prob=prob, text=text)

@app.route("/predict", methods=["POST"])
def predict():
    text = (request.get_json(silent=True) or {}).get("text", "")
    if not text.strip():
        return jsonify(error="No text provided"), 400
    p = float(model.predict_proba([text])[0][1])
    return jsonify(label="spam" if p >= 0.5 else "ham", spam_probability=round(p, 4))

@app.route("/health")
def health():
    return jsonify(status="ok")

if __name__ == "__main__":
    app.run(debug=False)
