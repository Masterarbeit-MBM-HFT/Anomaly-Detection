# 1. User Variables
_datasets_common_path = "./dummy_datasets/"
vin_path = _datasets_common_path + "supervised_learning/"
vin_name = "warnings_vin.csv"
sample_frac = 0.01 # for reducing the dataset size
sample_vin_name = "warnings_vin_sample.csv"

# 2. Import Libraries
import pandas as pd
import os

# 3. Functions

## 3.1 Return list of .csv files
def return_csv_lists(file_path):
    return os.listdir(f"{file_path}")

## 3.2 Merge .csv files into files
def merge_csv_files(_csv_files, file_path, file_name):
    df_final = pd.DataFrame()

    for file in _csv_files:
        df_t = pd.read_csv(f"{file_path}chunks/{file}", on_bad_lines="skip", sep=";")
        df_final = pd.concat([df_final, df_t])

    df_final.to_csv(f"{file_path}{file_name}", sep=";", index=False)

## 3.3 Reduce the dataset size for DATABRICKS memory problem :P
def save_stratified_sample(input_file_path:str, stratify_col:str, sample_frac:float, output_file_path: str):

    df_full = pd.read_csv(input_file_path, on_bad_lines="skip", sep=";")

    # Perform stratified sampling
    df_stratified = (
        df_full.groupby(stratify_col, group_keys=False)
          .apply(lambda x: x.sample(frac=sample_frac, random_state=42))
          .reset_index(drop=True)
    )
    
    # Save to CSV
    df_stratified.to_csv(output_file_path, index=False, sep=";")
    print(f"Stratified sample saved to {output_file_path}")

# 4. File Handling
_csv_files = return_csv_lists(vin_path+"chunks/")
_result = merge_csv_files(_csv_files, vin_path, vin_name)

# 5. Sampling
save_stratified_sample(vin_path+vin_name, "anomaly_flag", sample_frac, vin_path+sample_vin_name)

# 5. Final
print("Success!")