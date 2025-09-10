# Config constants for file paths and variables

import os

CHECK_CURRENT_OS_PATH = f"{os.listdir()}"
COMMON_PATH = "./anomaly_detection_final/data/"

INPUT_FILE_PATH = COMMON_PATH + "input/warnings_vin_sample_50000.csv"
OUTPUT_FILE_PATH = COMMON_PATH + "output/result.csv"
LOGGING_FILE_PATH = COMMON_PATH + "output/logs/result.log"
IMAGE_PATH = COMMON_PATH + "output/viz/"

TARGET_VARIABLE = "anomaly_flag"