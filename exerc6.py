import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice

features = [
    'LotArea',
    'YearBuilt',
    '1stFlrSF',
    '2ndFlrSF',
    'FullBath',
    'BedroomAbvGr',
    'TotRmsAbvGrd'
]

X = home_data[features]

train_X, val_X, train_y, val_y = train_test_split(
    X,
    y,
    random_state=1
)

iowa_model = DecisionTreeRegressor(random_state=1)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

iowa_model = DecisionTreeRegressor(
    max_leaf_nodes=100,
    random_state=1
)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

from learntools.core import binder
binder.bind(globals())

from learntools.machine_learning.ex6 import *

print("\nSetup complete")

rf_model = RandomForestRegressor(
    random_state=1
)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(
    val_y,
    rf_val_predictions
)

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))

step_1.check()