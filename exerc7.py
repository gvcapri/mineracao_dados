import pandas as pd
from sklearn.ensemble import RandomForestRegressor

train_data = pd.read_csv(r'C:\Users\guilh\OneDrive\Desktop\pessoal\codigos\python\train.csv')
test_data = pd.read_csv(r'C:\Users\guilh\OneDrive\Desktop\pessoal\codigos\python\test.csv')

features = [
    'LotArea',
    'YearBuilt',
    '1stFlrSF',
    '2ndFlrSF',
    'FullBath',
    'BedroomAbvGr',
    'TotRmsAbvGrd'
]

X = train_data[features]
y = train_data['SalePrice']
X_test = test_data[features]

rf_model = RandomForestRegressor(n_estimators=100, random_state=1)
rf_model.fit(X, y)

test_predictions = rf_model.predict(X_test)

output = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': test_predictions
})

output.to_csv('submission.csv', index=False)
print(output.head())