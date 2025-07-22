# ❤️ Heart Failure Prediction Web App (Flask + ML)

This project is a **Flask web application** that predicts the likelihood of heart failure based on patient medical data using a trained **Random Forest Classifier**.

---

## 🚀 Features

- Input form for 12 patient attributes.
- Predicts **Death Likely** or **Survival Likely**.
- Displays **probabilities** for survival and death.
- Backend built with **Flask**, frontend using **HTML/CSS**.
- Model trained using **Random Forest** for >80% accuracy.

---

## 📁 Project Structure

heart_failure_prediction/
├── app.py # Flask backend
├── model.pkl # Trained ML model
├── templates/
│ └── index.html # HTML frontend
└── README.md # This file 
  

  
---

## 📊 Input Features

The form expects the following 12 inputs:

1. `age` – Age of the patient (float)
2. `anaemia` – 1 if patient has anaemia, else 0
3. `creatinine_phosphokinase` – enzyme level (float)
4. `diabetes` – 1 if diabetic, else 0
5. `ejection_fraction` – % of blood leaving heart per beat
6. `high_blood_pressure` – 1 if yes, else 0
7. `platelets` – platelets count in blood
8. `serum_creatinine` – creatinine level
9. `serum_sodium` – sodium level
10. `sex` – 1 for male, 0 for female
11. `smoking` – 1 if patient smokes, else 0
12. `time` – follow-up period (days)

---

## 🧠 Model Training (in Python)

You can retrain the model using this code:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


🔧 How to Run the App
Step 1: Install Dependencies
pip install flask scikit-learn pandas
Step 2: Run Flask App
python app.py

🖥️ Sample Output
Prediction: Survival Likely ✅

Probability of Survival: 86.7%

Probability of Death: 13.3%

