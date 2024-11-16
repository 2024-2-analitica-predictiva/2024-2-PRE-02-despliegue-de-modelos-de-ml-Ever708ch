"""Build, deploy and access a model using sklearn"""

import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r"D:\ever\PhD\Predictiva\2024-2-PRE-02-despliegue-de-modelos-de-ml-Ever708ch\files\input\house_data.csv")

features = df[
              ["bedrooms",
               "bathrooms",
               "sqft_living",
               "sqft_lot",
               "floors",
               "waterfront",
               "condition",
               ]
               ]
target = df[["price"]]

estimator = LinearRegression()
estimator.fit(features, target)
with open("homework/house_predictor.pkl", "wb") as file:
    pickle.dump(estimator, file)