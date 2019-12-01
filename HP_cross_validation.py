import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

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

def get_score(X, y, n_estimators):
    """Return the average MAE over 3 CV folds of random forest model."""
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
        ('model', RandomForestRegressor(max_leaf_nodes=250, n_estimators=n_estimators, random_state=0))
    ])
    scores = -1 * cross_val_score(test_pipeline, X, y,
                                  cv=3,
                                  scoring='neg_mean_absolute_error')
    #print("Average MAE score:", scores.mean())
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

X_train, X_test, my_cols, numerical_cols,categorical_cols = columns_manipulate(X_full)

results = {}
for i in range(1, 9):
    results[50 * i] = get_score(X_train, y, 50 * i)
print(results)

best_n_estimators = get_best_n_estimators(results)

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

best_model = RandomForestRegressor(max_leaf_nodes=250, n_estimators=best_n_estimators, random_state=0)

my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', best_model)
])
# MODEL FITTING ON THE WHOLE TRAIN DATASET IN ORDER TO IMPROVE ACCURACY
my_pipeline.fit(X_full, y)
preds_test = my_pipeline.predict(X_test_full)

output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission_HP_CrossValidation.csv', index=False)
