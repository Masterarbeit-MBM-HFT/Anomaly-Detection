# models/train.py

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import logging

def train_logistic_regression(preprocessor, X_train, y_train, class_weight=None):
    logging.info("Training Logistic Regression model")
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight=class_weight)
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf)
    ])
    pipe.fit(X_train, y_train)
    return pipe

def train_random_forest(preprocessor, X_train, y_train, class_weight=None, n_estimators=200):
    logging.info("Training Random Forest model")
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight=class_weight)
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf)
    ])
    pipe.fit(X_train, y_train)
    return pipe