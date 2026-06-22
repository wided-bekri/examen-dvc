
import pandas as pd

import joblib

import json

import os

from sklearn.metrics import r2_score, mean_squared_error



X_test = pd.read_csv("data/X_test_scaled.csv")

y_test = pd.read_csv("data/y_test.csv").values.ravel()

model = joblib.load("models/gbr_model.pkl")



predictions = model.predict(X_test)



os.makedirs("data", exist_ok=True)

os.makedirs("metrics", exist_ok=True)



pd.DataFrame(predictions, columns=["prediction"]).to_csv("data/prediction.csv", index=False)



scores = {

    "r2": r2_score(y_test, predictions),

    "mse": mean_squared_error(y_test, predictions)

}

with open("metrics/scores.json", "w") as f:

    json.dump(scores, f)



print("Evaluation done:", scores)

