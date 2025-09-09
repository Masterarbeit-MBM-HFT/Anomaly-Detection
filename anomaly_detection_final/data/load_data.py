# data/load_data.py

import pandas as pd
import logging

def load_data(path, sep=';'):
    logging.info(f"Loading data from {path}")
    try:
        df = pd.read_csv(path, sep=sep)
        df["reporting_date"] = pd.to_datetime(df["reporting_date"], errors="coerce")
        logging.info("Converted reporting_date to datetime")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise
