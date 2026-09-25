import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

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


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes,
        random_state=1
    )
    
    model.fit(train_X, train_y)
    
    predictions = model.predict(val_X)
    
    mae = mean_absolute_error(val_y, predictions)
    
    return mae


candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

scores = {}

for max_leaf_nodes in candidate_max_leaf_nodes:
    scores[max_leaf_nodes] = get_mae(
        max_leaf_nodes,
        train_X,
        val_X,
        train_y,
        val_y
    )


best_tree_size = min(scores, key=scores.get)

print("MAE para cada tamanho:")
print(scores)

print("Melhor tamanho da árvore:", best_tree_size)
print("Melhor MAE:", scores[best_tree_size])


final_model = DecisionTreeRegressor(
    max_leaf_nodes=best_tree_size,
    random_state=1
)

final_model.fit(X, y)