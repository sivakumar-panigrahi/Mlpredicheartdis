from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read input data from form
        features = [
            float(request.form['age']),
            float(request.form['anaemia']),
            float(request.form['creatinine_phosphokinase']),
            float(request.form['diabetes']),
            float(request.form['ejection_fraction']),
            float(request.form['high_blood_pressure']),
            float(request.form['platelets']),
            float(request.form['serum_creatinine']),
            float(request.form['serum_sodium']),
            float(request.form['sex']),
            float(request.form['smoking']),
            float(request.form['time'])
        ]

        # Predict class and probabilities
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]  # [prob_survival, prob_death]

        result = "Death likely" if prediction == 1 else "Survival likely"

        return render_template(
            'index.html',
            prediction=result,
            prob_survival=round(probabilities[0] * 100, 2),
            prob_death=round(probabilities[1] * 100, 2),
            show_graph=True
        )

    except Exception as e:
        return render_template('index.html', prediction="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
