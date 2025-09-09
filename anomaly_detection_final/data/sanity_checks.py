# data/sanity_checks.py

import logging
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
import math

def basic_sanity_checks(df, target_variable):
    logging.info(f"Dataframe shape: {df.shape}")
    logging.info(f"Dataframe info:\n{df.info()}")
    logging.info(f"5 Number statistics of {target_variable} is: {df[target_variable].describe()}")

    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    logging.info(f"Constant columns: {constant_cols}")

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    high_card = [c for c in cat_cols if df[c].nunique() > 100]
    logging.info(f"High-cardinality categoricals: {high_card}")

    feature_info = pd.DataFrame({
        "dtype": df.dtypes,
        "n_missing": df.isna().sum(),
        "% missing": (df.isna().sum() / len(df) * 100).round(2),
        "n_unique": df.nunique(),
    })
    logging.info(f"Feature Information:\n{feature_info}")

    n_dupes = df.duplicated().sum()
    logging.info(f"Number of duplicate rows: {n_dupes}")

    df_vin_contradict = df[df["ctt_vin"].isna() & (df["anomaly_description"].str.contains("No anomaly detected", na=False))]
    logging.info(f"Records with missing VIN but 'No anomaly detected': {len(df_vin_contradict)}")


def analyze_class_imbalance(df, target_variable):
    logging.info("Analyzing class imbalance")
    class_counts = df[target_variable].value_counts()
    logging.info(f"Class counts: {class_counts.to_dict()}")

    n0, n1 = class_counts.values
    cir = max(n0, n1) / min(n0, n1)
    logging.info(f"Class imbalance ratio: {cir}")

    n_total = len(df)
    n_pos = n1
    prevalence = n_pos / n_total
    logging.info(f"Prevalence: {prevalence}")

    ci_low, ci_high = proportion_confint(count=n_pos, nobs=n_total, method="wilson")
    logging.info(f"Wilson 95% CI for prevalence: low: {ci_low}, high: {ci_high}")

    SE = (prevalence * (1 - prevalence) / n_total) ** 0.5
    logging.info(f"Standard Error: {SE}")


def required_n_for_proportion(m=0.05, confidence=0.95, p=None):
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence) / 2)
    if p is None:
        p = 0.5
    n_needed = math.ceil((z**2) * p * (1 - p) / (m**2))
    return n_needed


def calculate_required_sample_size(df, target_variable, margin_of_error=0.05, confidence=0.95):
    n_needed = required_n_for_proportion(m=margin_of_error, confidence=confidence)
    p_hat = df[target_variable].mean()
    n_pos_needed = math.ceil(n_needed * p_hat)
    n_neg_needed = n_needed - n_pos_needed
    logging.info(f"Required sample size total: {n_needed}")
    logging.info(f"Required positives: {n_pos_needed}")
    logging.info(f"Required negatives: {n_neg_needed}")
    return n_needed, n_pos_needed, n_neg_needed