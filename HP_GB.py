import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
#import matplotlib.pyplot as plt

def get_best_n_estimators(candidate_best_n_estimators):
    best = min(candidate_best_n_estimators, key=candidate_best_n_estimators.get)
    print("best_n_estmators = %d" % (best))
    return (best)

def columns_manipulate(X):
    # Select numerical columns
    numerical_cols = [cname for cname in X.columns if
                      X[cname].dtype in ['int64', 'float64']]

    # Select categorical columns with relatively low cardinality
    categorical_cols = [cname for cname in X.columns if
                        X[cname].nunique() < 10 and
                        X[cname].dtype == "object"]

    # Keep selected columns only
    my_cols = categorical_cols + numerical_cols
    X_train = X[my_cols].copy()
    X_test = X_test_full[my_cols].copy()
    return (X_train, X_test, my_cols, numerical_cols, categorical_cols)

def get_score(X, y, n_est):
    """Return the average MAE over 6 CV folds of random forest model."""
    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='constant')

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    test_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model',  GradientBoostingRegressor(n_estimators=n_est, learning_rate=0.04))
    ])
    scores = -1 * cross_val_score(test_pipeline, X, y,
                                  cv=6,
                                  scoring='neg_mean_absolute_error')
    print("Average MAE score:", scores.mean())
    return (scores.mean())

train_file_path = (r'C:\PY_workdir\HomePrices\train.csv')
test_file_path = (r'C:\PY_workdir\HomePrices\test.csv')

# Read the data
X_full = pd.read_csv(train_file_path, index_col='Id')
X_test_full = pd.read_csv(test_file_path, index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

#remove noise from train dataset
X_train, X_test, my_cols, numerical_cols,categorical_cols = columns_manipulate(X_full)

candidates_estim=[1200, 1300, 1400, 1500, 1600, 1700]
results = {}
for i in candidates_estim:
    results[i] = get_score(X_train, y, i)
print(results)

best_n_estimators = min(results, key=results.get)

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

#Best Average MAE score found: 15350.047150300374 with n_estimators=1300 learning_rate=0.04 and cv 6 in cross_val_score
model = GradientBoostingRegressor(n_estimators=best_n_estimators, learning_rate=0.04)

my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

scores = -1 * cross_val_score(my_pipeline, X_train, y,
                                 cv=6,
                                scoring='neg_mean_absolute_error')
print("Average MAE score:", scores.mean())

# MODEL FITTING ON THE WHOLE TRAIN DATASET IN ORDER TO IMPROVE ACCURACY

my_pipeline.fit(X_full, y)
preds_test = my_pipeline.predict(X_test_full)

output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission_HP_GB.csv', index=False)
