import pandas as pd
from sklearn.tree import DecisionTreeRegressor

caminho= r'C:\Users\guilh\OneDrive\Desktop\pessoal\codigos\python\train.csv'

home_data = pd.read_csv(caminho)

y = home_data.SalePrice

colunas = [
    'LotArea',
    'YearBuilt',
    '1stFlrSF',
    '2ndFlrSF',
    'FullBath',
    'BedroomAbvGr',
    'TotRmsAbvGrd'
]

X = home_data[colunas]

model = DecisionTreeRegressor()

model.fit(X, y)

print("predição primeira amostra:", model.predict(X.head()))
print("valores alvo reais para essas casas:", y.head().tolist())

#from learntools.core import binder
#binder.bind(globals())

#from learntools.machine_learning.ex4 import *

#print("Setup Complete")