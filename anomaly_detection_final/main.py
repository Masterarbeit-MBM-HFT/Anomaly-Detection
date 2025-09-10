# main.py

import logging
from config.constants import CHECK_CURRENT_OS_PATH, INPUT_FILE_PATH, OUTPUT_FILE_PATH, LOGGING_FILE_PATH, IMAGE_PATH, TARGET_VARIABLE
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
from sklearn.model_selection import StratifiedShuffleSplit

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

    # Iterative Testing
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    results_list = []

    for i, (train_index, test_index) in enumerate(sss.split(X, y), start=1):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        plot_class_distribution(y_train, y_test, IMAGE_PATH+f'class_distribution_iter_{i}.png')

        preprocessor = build_preprocessor(X_train)

        lr_model = train_logistic_regression(preprocessor, X_train, y_train)
        rf_model = train_random_forest(preprocessor, X_train, y_train)

        lr_model_balanced = train_logistic_regression(preprocessor, X_train, y_train, class_weight="balanced")
        rf_model_balanced = train_random_forest(preprocessor, X_train, y_train, class_weight="balanced")

        lr_metrics = evaluate(lr_model, X_train, X_test, y_train, y_test)
        rf_metrics = evaluate(rf_model, X_train, X_test, y_train, y_test)

        lr_bal_metrics = evaluate(lr_model_balanced, X_train, X_test, y_train, y_test)
        rf_bal_metrics = evaluate(rf_model_balanced, X_train, X_test, y_train, y_test)

        plot_roc_curves([("Logistic Regression", lr_model), ("Random Forest", rf_model)], 
                        X_test, y_test, 
                        IMAGE_PATH+f'roc_curves_iter_{i}.png')
        plot_roc_curves([("Logistic Regression (Class Balanced)", lr_model_balanced), ("Random Forest (Class Balanced)", rf_model_balanced)], 
                        X_test, y_test, 
                        IMAGE_PATH+f'roc_curves_balanced_iter_{i}.png')


        # Add iteration number to metrics before appending
        for metrics, model_name in zip(
            [lr_metrics, lr_bal_metrics, rf_metrics, rf_bal_metrics],
            [
                "Logistic Regression (Before Class Balance)",
                "Logistic Regression (After Class Balance)",
                "Random Forest (Before Class Balance)",
                "Random Forest (After Class Balance)",
            ],
        ):
            metrics["model"] = model_name
            metrics["iteration"] = i
            results_list.append(metrics)


    results_df = pd.DataFrame(results_list)
    results_df.to_csv(OUTPUT_FILE_PATH, index=False)

if __name__ == '__main__':
    main()