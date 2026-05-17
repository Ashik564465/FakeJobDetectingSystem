from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load models
lr_model = pickle.load(open("lr_model.pkl", "rb"))
nb_model = pickle.load(open("nb_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    lr_prob = None
    nb_prob = None
    risk = None

    if request.method == "POST":
        text = request.form["job_desc"]

        vec = vectorizer.transform([text])

        # Probabilities
        lr_prob = lr_model.predict_proba(vec)[0][1]
        nb_prob = nb_model.predict_proba(vec)[0][1]

        # Ensemble Logic
        if lr_prob >= 0.2:
            final_pred = 1
        elif nb_prob >= 0.6:
            final_pred = 1
        else:
            final_pred = 0

        # Result
        result = "Fake Job ⚠️" if final_pred == 1 else "Real Job ✅"
        confidence = round(max(lr_prob, nb_prob) * 100, 2)

        # Risk Level
        if confidence > 80:
            risk = "High Risk 🔴"
        elif confidence > 50:
            risk = "Moderate Risk 🟠"
        else:
            risk = "Low Risk 🟢"

    return render_template("index.html",
                           result=result,
                           confidence=confidence,
                           lr_prob=lr_prob,
                           nb_prob=nb_prob,
                           risk=risk)

if __name__ == "__main__":
    app.run(debug=True)