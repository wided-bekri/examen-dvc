
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

import joblib

import os



X_train = pd.read_csv("data/X_train_scaled.csv")

y_train = pd.read_csv("data/y_train.csv").values.ravel()

best_params = joblib.load("models/best_params.pkl")



model = GradientBoostingRegressor(**best_params, random_state=42)

model.fit(X_train, y_train)



os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/gbr_model.pkl")

print("Training done.")

