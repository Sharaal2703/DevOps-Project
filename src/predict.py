import joblib
import pandas as pd

model = joblib.load("model.pkl")


def predict(features):

    data = pd.DataFrame([features])

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0].max()

    return int(prediction), float(probability)