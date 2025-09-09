# preprocessing/pipelines.py

import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def build_preprocessor(X_train):
    numeric_features = X_train.select_dtypes(include=["int64", "float64", "number"]).columns
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns

    logging.info(f"Numeric features: {list(numeric_features)}")
    logging.info(f"Categorical features: {list(categorical_features)}")

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor
