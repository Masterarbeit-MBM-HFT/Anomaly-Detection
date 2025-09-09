# data/feature_selection.py

import logging
import numpy as np

def feature_selection(df, target_variable, ids_to_keep=['ctt_cms_contract_number', 'ctt_vin']):
    logging.info("Calculating correlation with target")
    cor_target = df.corr(numeric_only=True)[target_variable].abs().sort_values(ascending=False)
    logging.info(f"Correlation with target variable '{target_variable}':\n{cor_target}")

    drop_cols = [col for col in df.columns if df[col].nunique() == 1]
    leakage_candidates = ['anomaly_description']
    drop_cols += [col for col in leakage_candidates if col in df.columns]
    drop_cols = [col for col in drop_cols if col not in ids_to_keep]
    logging.info(f"Dropping columns: {drop_cols}")

    df_model = df.drop(columns=drop_cols)

    numeric_feats = df_model.select_dtypes(include=[np.number]).columns.tolist()
    if target_variable in numeric_feats:
        numeric_feats.remove(target_variable)
    corr_matrix = df_model[numeric_feats].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [col for col in upper.columns if any(upper[col] > 0.95)]
    logging.info(f"Dropping highly correlated columns: {high_corr}")
    df_model = df_model.drop(columns=high_corr)

    logging.info(f"Final dataframe shape for modeling: {df_model.shape}")
    return df_model
