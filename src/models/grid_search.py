
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV

import joblib

import os



X_train = pd.read_csv("data/X_train_scaled.csv")

y_train = pd.read_csv("data/y_train.csv").values.ravel()



param_grid = {

    "n_estimators": [100, 200],

    "max_depth": [3, 5],

    "learning_rate": [0.05, 0.1]

}



gbr = GradientBoostingRegressor(random_state=42)

grid_search = GridSearchCV(gbr, param_grid, cv=3, scoring="r2", n_jobs=-1)

grid_search.fit(X_train, y_train)



os.makedirs("models", exist_ok=True)

joblib.dump(grid_search.best_params_, "models/best_params.pkl")

print("Best params:", grid_search.best_params_)

