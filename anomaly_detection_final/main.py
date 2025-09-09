# main.py

import logging
from config.constants import CHECK_CURRENT_OS_PATH, INPUT_FILE_PATH, OUTPUT_FILE_PATH, LOGGING_FILE_PATH, TARGET_VARIABLE
from utils.logging_setup import setup_logging
from data.load_data import load_data
from data.sanity_checks import basic_sanity_checks, analyze_class_imbalance, calculate_required_sample_size
from data.feature_selection import feature_selection
from preprocessing.pipelines import build_preprocessor
from sklearn.model_selection import train_test_split
from models.train import train_logistic_regression, train_random_forest
from models.evaluate import evaluate
from utils.visualization import plot_class_distribution, plot_roc_curves
import pandas as pd

def main():
    # Test
    print(f"The current working directory is: {CHECK_CURRENT_OS_PATH}")

    setup_logging(LOGGING_FILE_PATH)
    logging.info("Starting anomaly detection script")    

    df = load_data(INPUT_FILE_PATH)
    basic_sanity_checks(df, TARGET_VARIABLE)
    analyze_class_imbalance(df, TARGET_VARIABLE)
    calculate_required_sample_size(df, TARGET_VARIABLE)

    df_model = feature_selection(df, TARGET_VARIABLE)

    X = df_model.drop(columns=[TARGET_VARIABLE])
    y = df_model[TARGET_VARIABLE]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    plot_class_distribution(y_train, y_test)

    preprocessor = build_preprocessor(X_train)

    lr_model = train_logistic_regression(preprocessor, X_train, y_train)
    rf_model = train_random_forest(preprocessor, X_train, y_train)

    lr_metrics = evaluate(lr_model, X_train, X_test, y_train, y_test)
    rf_metrics = evaluate(rf_model, X_train, X_test, y_train, y_test)

    results_df = pd.DataFrame([lr_metrics, rf_metrics])
    results_df["model"] = ["Logistic Regression", "Random Forest"]
    results_df.to_csv(OUTPUT_FILE_PATH, index=False)

    plot_roc_curves([("Logistic Regression", lr_model), ("Random Forest", rf_model)], X_test, y_test)

if __name__ == '__main__':
    main()