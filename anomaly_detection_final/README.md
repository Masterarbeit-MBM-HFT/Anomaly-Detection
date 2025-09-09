# Anomaly Detection with Logistic Regression and Random Forest

## Project Overview
This project implements an end-to-end supervised anomaly detection pipeline using Logistic Regression and Random Forest classifiers. It is designed for tabular data with imbalanced binary targets (anomaly flags). The project covers data loading, sanity checks, feature selection, preprocessing, model training, evaluation, and result visualization, organized following clean architecture principles for maintainability and extensibility.

## Features
* Modular codebase separating configuration, data handling, preprocessing, models, evaluation, and utilities.
* Robust data sanity checks including class imbalance analysis and feature correlation pruning.
* Preprocessing pipelines handling numeric and categorical features with imputation and scaling/encoding.
* Model training with class imbalance handling using scikit-learn pipelines.
* Evaluation with multiple metrics: accuracy, ROC AUC, precision, recall, F1-score, balanced accuracy.
* Visualization of class distributions and ROC curves.
* Logging throughout the pipeline for traceability and debugging.

## Installation

1. Clone the repository:

```bash
git clone <repo_url>
cd anomaly_detection
Create and activate a Python environment (recommended):
```

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:
```

```bash
pip install -r requirements.txt
```

Dependencies include pandas, numpy, scikit-learn, matplotlib, seaborn, statsmodels, and scipy.

## Usage

Run the main pipeline with:

```bash
python main.py
```

The script will:
* Load and preprocess the data from the configured input path.
* Perform sanity checks and analyze class imbalance.
* Select features and split data into train/test sets.
* Train Logistic Regression and Random Forest models.
* Evaluate and log metrics, and save results to the output path.
* Show visualizations for class distribution and ROC curves.

## Code Structure

``config/constants.py``: Configuration variables such as file paths and target column name.
``data/``: Data loading, sanity checks, and feature selection modules.
``preprocessing/``: Preprocessing pipelines for numeric and categorical data.
``models/``: Training and evaluation logic for classifiers.
``utils/``: Logging setup and visualization utilities.
``main.py``: Orchestrates the pipeline flow with calls to modular functions.

## Data

The project expects input CSV data with semicolon separators and a date column named ``reporting_date``. The target variable column is ``anomaly_flag``. Adjust file paths and column names in ``config/constants.py`` as needed.

## Results

1. Evaluation metrics and model comparison results are saved as CSV in the configured output directory.
2. Log files with detailed execution info are saved under the logging directory.
3. Visual plots are displayed during execution to provide insights about data distribution and model performance.

## Future Work

* Extend to include additional anomaly detection algorithms.
* Add hyperparameter tuning with cross-validation.
* Integrate logging and visualization with dashboard tools for real-time monitoring.
* Support for streaming or time-series anomaly detection.