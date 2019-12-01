import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def columns_manipulate(X, shape):
    # Select numerical columns
    numerical_cols = [cname for cname in X.columns if
                      X[cname].dtype in ['int64', 'float64']]

    # Select categorical columns with relatively low cardinality
    categorical_cols = [cname for cname in X.columns if
                        X[cname].nunique() < 10 and
                        X[cname].dtype == "object"]

    if shape == "splitted":
        # Keep selected columns only
        my_cols = categorical_cols + numerical_cols
        X_train = X[my_cols].copy()
        X_valid = X_full_valid[my_cols].copy()
        X_test = X_test_full[my_cols].copy()
        return (X_train, X_valid, X_test,my_cols,numerical_cols,categorical_cols)
    elif shape == "full":
         # Keep selected columns only
         my_cols = categorical_cols + numerical_cols
         X_train = X[my_cols].copy()
         X_test = X_test_full[my_cols].copy()
         return (X_train, X_test, my_cols, numerical_cols, categorical_cols)

train_file_path = (r'C:\PY_workdir\HomePrices\train.csv')
test_file_path = (r'C:\PY_workdir\HomePrices\test.csv')

# Read the data
X_full = pd.read_csv(train_file_path, index_col='Id')
X_test_full = pd.read_csv(test_file_path, index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

X_full_train, X_full_valid, train_y, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)

X_train, X_valid, X_test, my_cols, numerical_cols,categorical_cols = columns_manipulate(X_full_train, "splitted")

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
#define model
model = RandomForestRegressor(max_leaf_nodes=250, n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, train_y)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)

# NOW PREPROCESSING AND MODEL FITTING ON THE WHOLE TRAIN DATASET IN ORDER TO IMPROVE ACCURACY
X_train, X_test, my_cols, numerical_cols_full, categorical_cols_full = columns_manipulate(X_full, "full")

# Bundle preprocessing for numerical and categorical data ON FULL TRAIN DATASET
preprocessor_on_full_train = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols_full),
        ('cat', categorical_transformer, categorical_cols_full)
    ])

my_pipeline_on_full_train = Pipeline(steps=[('preprocessor', preprocessor_on_full_train),
                              ('model', model)
                             ])

my_pipeline_on_full_train.fit(X_full, y)

preds_test = my_pipeline_on_full_train.predict(X_test_full)

output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission_HP_Pipelines.csv', index=False)
#MAE PREDICTIONS = 16257.45528