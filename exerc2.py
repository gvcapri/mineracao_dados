import pandas as pd

caminho = r"/kaggle/input/home-data-for-ml-course/train.csv"
dados = pd.read_csv(caminho)

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex2 import *

print(dados.columns)

y = dados["SalePrice"]

step_1.check()

feature_names = [
    "LotArea",
    "YearBuilt",
    "1stFlrSF",
    "2ndFlrSF",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
]

X = dados[feature_names]

step_2.check()

print(X.describe())
print(X.head())

from sklearn.tree import DecisionTreeRegressor

iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(X, y)

step_3.check()

predictions = iowa_model.predict(X)

print(predictions)

step_4.check()

